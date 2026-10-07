# scene/gaussian_model.py

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation, _axis_angle_rotation, matrix_to_quaternion


class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.sigmoid
        self.m_activation = torch.softmax

    def __init__(self, sh_degree: int, polynomial_degree: int = 1, frames: int = 0, use_dff: bool = False):
        self.active_sh_degree = 0
        self.polynomial_degree = polynomial_degree
        self.max_sh_degree = sh_degree
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)

        # Segmentation head parameters (initialized via init_seg_head_from_img)
        self._features_dc_seg = torch.empty(0)
        self._features_rest_seg = torch.empty(0)
        self._opacity_seg = torch.empty(0)

        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.m = torch.empty(0)
        self._w1 = torch.empty(0)
        self.sigma = torch.empty(0)
        self.time_func = torch.empty(0)
        self.use_dff = use_dff
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.eps_s3 = 1e-8
        self.frames = frames
        self.setup_functions()

   
    # Small helpers

    def _has_param_group(self, name: str) -> bool:
        if self.optimizer is None:
            return False
        return any(g.get("name", "") == name for g in self.optimizer.param_groups)

    def has_seg_head(self) -> bool:
        if self._xyz is None or self._xyz.numel() == 0:
            return False
        if self._opacity_seg is None or self._features_dc_seg is None or self._features_rest_seg is None:
            return False
        if not isinstance(self._opacity_seg, torch.Tensor) or self._opacity_seg.numel() == 0:
            return False
        n = self._xyz.shape[0]
        return (
            self._opacity_seg.shape[0] == n
            and self._features_dc_seg.shape[0] == n
            and self._features_rest_seg.shape[0] == n
        )


    # Checkpointing 

    def capture(self):
        return {
            "version": 2,
            "active_sh_degree": self.active_sh_degree,
            "_xyz": self._xyz,
            "_features_dc": self._features_dc,
            "_features_rest": self._features_rest,
            "_scaling": self._scaling,
            "_rotation": self._rotation,
            "_opacity": self._opacity,
            "max_radii2D": self.max_radii2D,
            "xyz_gradient_accum": self.xyz_gradient_accum,
            "denom": self.denom,
            "optimizer_state": self.optimizer.state_dict() if self.optimizer is not None else None,
            "spatial_lr_scale": self.spatial_lr_scale,

            # temporal / motion-related parameters
            "m": self.m,
            "sigma": self.sigma,
            "_w1": self._w1,
            "time_func": self.time_func,

            # polynomial degree
            "polynomial_degree": self.polynomial_degree,

            # segmentation head (may be empty if not initialized)
            "_opacity_seg": self._opacity_seg,
            "_features_dc_seg": self._features_dc_seg,
            "_features_rest_seg": self._features_rest_seg,
        }

    def restore(self, model_args, training_args=None, load_optimizer=True):

        if isinstance(model_args, (tuple, list)):
            (self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            xyz_gradient_accum,
            denom,
            opt_dict,
            self.spatial_lr_scale) = model_args

            self.xyz_gradient_accum = xyz_gradient_accum
            self.denom = denom


            dev = self._opacity.device
            self._opacity_seg = torch.empty(0, device=dev)
            self._features_dc_seg = torch.empty(0, device=dev)
            self._features_rest_seg = torch.empty(0, device=dev)

            if training_args is not None:
                self.training_setup(training_args)
                if load_optimizer and opt_dict is not None:
                    self.optimizer.load_state_dict(opt_dict)
            else:
                self.optimizer = None
            return


        assert isinstance(model_args, dict), "Unexpected checkpoint format"

        self.active_sh_degree = model_args["active_sh_degree"]
        self._xyz = model_args["_xyz"]
        self._features_dc = model_args["_features_dc"]
        self._features_rest = model_args["_features_rest"]
        self._scaling = model_args["_scaling"]
        self._rotation = model_args["_rotation"]
        self._opacity = model_args["_opacity"]
        self.max_radii2D = model_args["max_radii2D"]
        self.xyz_gradient_accum = model_args["xyz_gradient_accum"]
        self.denom = model_args["denom"]
        self.spatial_lr_scale = model_args["spatial_lr_scale"]

        self.m = model_args["m"]
        self.sigma = model_args["sigma"]
        self._w1 = model_args["_w1"]
        self.time_func = model_args["time_func"]

        self.polynomial_degree = model_args.get("polynomial_degree", None)
        if self.polynomial_degree is None:
            self.polynomial_degree = self._w1.shape[-1] // 2

        # seg head: default to EMPTY tensors=
        dev = self._opacity.device
        self._opacity_seg = model_args.get("_opacity_seg", torch.empty(0, device=dev))
        self._features_dc_seg = model_args.get("_features_dc_seg", torch.empty(0, device=dev))
        self._features_rest_seg = model_args.get("_features_rest_seg", torch.empty(0, device=dev))

        opt_state = model_args.get("optimizer_state", None)

        if training_args is not None:
            self.training_setup(training_args)
            if load_optimizer and opt_state is not None:
                self.optimizer.load_state_dict(opt_state)
        else:
            self.optimizer = None

    # Accessors

    @property
    def get_scaling(self):
        s3 = torch.ones(self._scaling.shape[0], 1).cuda() * self.eps_s3
        s12 = self.scaling_activation(self._scaling[:, [0, -1]])
        s = torch.cat([s12[:, 0].unsqueeze(1), s3, s12[:, -1].unsqueeze(1)], dim=1)
        return s

    @property
    def get_rotation_old(self):
        angle = self.rotation_activation(self._rotation.squeeze()) * 2 * np.pi
        rot_matrix = _axis_angle_rotation('Y', angle)
        return matrix_to_quaternion(rot_matrix)

    @property
    def get_rotation(self):
        angle = (self.rotation_activation(self._rotation.squeeze()) * 2 * torch.pi) / 2
        w = torch.cos(angle)
        x = torch.zeros_like(w)
        y = torch.sin(angle)
        z = torch.zeros_like(w)
        return torch.stack((w, x, y, z), dim=-1)

    @property
    def get_m(self):
        return torch.sigmoid(self.m)

    @property
    def get_sigma(self):
        return self.scaling_activation(self.sigma)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_time(self):
        if self.use_dff:
            return torch.nn.functional.softmax(self.time_func.squeeze(), dim=0)
        else:
            return self.time_func

    # head-aware feature getter
    def get_features_head(self, head: str = "img"):
        if head == "img":
            features_dc = self._features_dc
            features_rest = self._features_rest
        elif head == "seg":
            features_dc = self._features_dc_seg
            features_rest = self._features_rest_seg
        else:
            raise ValueError(f"Unknown head: {head}")
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_features(self):
        return self.get_features_head("img")

    # head-aware opacity getter
    def get_opacity_head(self, head: str = "img"):
        if head == "img":
            base = self._opacity
        elif head == "seg":
            base = self._opacity_seg
        else:
            raise ValueError(f"Unknown head: {head}")
        return self.opacity_activation(base)

    @property
    def get_opacity(self):
        return self.get_opacity_head("img")

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

  
    # Segmentation head helpers

    def init_seg_head_from_img(self):
        """
        Create seg-head tensors by copying current img head tensors.
        """
        self._opacity_seg = nn.Parameter(self._opacity.detach().clone().requires_grad_(True))
        self._features_dc_seg = nn.Parameter(self._features_dc.detach().clone().requires_grad_(True))
        self._features_rest_seg = nn.Parameter(self._features_rest.detach().clone().requires_grad_(True))

    def freeze_geometry_and_img_head(self):
        """
        Stage-2 helper: freeze everything except seg head.
        """
        param_list = [
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._opacity,
            self._scaling,
            self._rotation,
            self.m,
            self.sigma,
            self._w1,
        ]
        if isinstance(self.time_func, torch.nn.Parameter):
            param_list.append(self.time_func)

        for p in param_list:
            if isinstance(p, (torch.nn.Parameter, torch.Tensor)):
                p.requires_grad_(False)

    def seg_head_parameters(self):
        """
        Returns seg-head params (used in seg_head_only mode).

        """
        params = []
        if isinstance(self._opacity_seg, torch.nn.Parameter):
            params.append(self._opacity_seg)
        if isinstance(self._features_dc_seg, torch.nn.Parameter):
            params.append(self._features_dc_seg)
        if isinstance(self._features_rest_seg, torch.nn.Parameter):
            params.append(self._features_rest_seg)
        return params

  
    # Initialization from point cloud

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots = torch.logit(torch.rand((fused_point_cloud.shape[0], 1), device="cuda"))
        w1 = torch.rand((fused_point_cloud.shape[0], 2 * self.polynomial_degree), device="cuda") * 2.0 - 1.0
        m = torch.logit(torch.rand((fused_point_cloud.shape[0], 1), device="cuda"))
        sigma = torch.log(torch.rand((fused_point_cloud.shape[0], 1, 1), device="cuda") * 0.99 + 0.01)

        if self.use_dff:
            print("USING DFF")
            time_func = torch.ones(self.frames - 1, device="cuda")
            self.time_func = nn.Parameter(time_func.unsqueeze(-1).requires_grad_(True))
        else:
            print("NOT USING DFF")
            self.time_func = torch.ones(self.frames, device="cuda") / self.frames
            print(self.time_func)

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
        fused_point_cloud[:, 1] = 0

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        self.m = nn.Parameter(m.requires_grad_(True))
        self.sigma = nn.Parameter(sigma.requires_grad_(True))
        self._w1 = nn.Parameter(w1.requires_grad_(True))




    # Optimizer / training setup



    def training_setup(self, training_args):
        """
        Build optimizer param groups for ALL per-Gaussian tensors.
        """

        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.m_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.m_denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")


        if self.has_seg_head():
            if not isinstance(self._features_dc_seg, nn.Parameter):
                self._features_dc_seg = nn.Parameter(
                    self._features_dc_seg.detach().clone().requires_grad_(True)
                )
            if not isinstance(self._features_rest_seg, nn.Parameter):
                self._features_rest_seg = nn.Parameter(
                    self._features_rest_seg.detach().clone().requires_grad_(True)
                )
            if not isinstance(self._opacity_seg, nn.Parameter):
                self._opacity_seg = nn.Parameter(
                    self._opacity_seg.detach().clone().requires_grad_(True)
                )

        # ---- Base (img) + geometry + temporal groups ----
        l = [
            {"params": [self._xyz],           "lr": training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {"params": [self._features_dc],   "lr": training_args.feature_lr,                               "name": "f_dc"},
            {"params": [self._features_rest], "lr": training_args.feature_lr / 20.0,                        "name": "f_rest"},
            {"params": [self._opacity],       "lr": training_args.opacity_lr,                               "name": "opacity"},
            {"params": [self._scaling],       "lr": training_args.scaling_lr,                               "name": "scaling"},
            {"params": [self._rotation],      "lr": training_args.rotation_lr,                              "name": "rotation"},
            {"params": [self.m],              "lr": 0.001,                                                  "name": "m"},
            {"params": [self.sigma],          "lr": 0.001,                                                  "name": "sigma"},
            {"params": [self._w1],            "lr": 0.001,                                                  "name": "w1"},
        ]

        if self.use_dff:
            # time_func is global (not per-Gaussian); densification code skips this group by name
            l.append({"params": [self.time_func], "lr": 0.001, "name": "time_func"})

        if self.has_seg_head():
            l.extend([
                {"params": [self._features_dc_seg],   "lr": training_args.feature_lr,        "name": "f_dc_seg"},
                {"params": [self._features_rest_seg], "lr": training_args.feature_lr / 20.0, "name": "f_rest_seg"},
                {"params": [self._opacity_seg],       "lr": training_args.opacity_lr,        "name": "opacity_seg"},
            ])

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.spatial_lr_scale,
            lr_final=training_args.position_lr_final * self.spatial_lr_scale,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps,
        )



    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        for i in range(self.m.shape[1]):
            l.append('m_{}'.format(i))
        for i in range(self.sigma.shape[1]):
            l.append('sigma_{}'.format(i))
        for i in range(self._w1.shape[1]):
            l.append('w1_{}'.format(i))
        return l

    def save_ply(self, path):
        self._save_ply(path)

    def _save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self.get_xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        m = self.m.detach().cpu().numpy()
        w1 = self._w1.detach().cpu().numpy()
        sigma = self.sigma.squeeze(-1).detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation, m, sigma, w1), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')

        dtype_full = []
        for i in range(self.time_func.shape[0]):
            dtype_full.append('time_func_{}'.format(i))
        dtype_full = [(attribute, 'f4') for attribute in dtype_full]

        time_arr = np.empty(1, dtype=dtype_full)
        time_arr[:] = list(map(tuple, self.time_func.unsqueeze(0).squeeze(-1).detach().cpu().numpy()))
        time_el = PlyElement.describe(time_arr, 'time')
        PlyData([el, time_el]).write(path)


    def reset_opacity(self, reset_seg: bool = True):
        op_img_new = inverse_sigmoid(torch.min(
            self.get_opacity_head("img"),
            torch.ones_like(self.get_opacity_head("img")) * 0.01
        ))
        opt_img = self.replace_tensor_to_optimizer(op_img_new, "opacity")
        if "opacity" in opt_img:
            self._opacity = opt_img["opacity"]

        if reset_seg and self.has_seg_head():
            if self.optimizer is not None and any(g.get("name", "") == "opacity_seg" for g in self.optimizer.param_groups):
                op_seg_new = inverse_sigmoid(torch.min(
                    self.get_opacity_head("seg"),
                    torch.ones_like(self.get_opacity_head("seg")) * 0.01
                ))
                opt_seg = self.replace_tensor_to_optimizer(op_seg_new, "opacity_seg")
                if "opacity_seg" in opt_seg:
                    self._opacity_seg = opt_seg["opacity_seg"]

    def load_ply(self, path):
        self._load_ply(path)

    def _load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])), axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3

        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        features_extra = features_extra.reshape(
            (features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1)
        )

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        m_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("m")]
        m_names = sorted(m_names, key=lambda x: int(x.split('_')[-1]))
        m = np.zeros((xyz.shape[0], len(m_names)))
        for idx, attr_name in enumerate(m_names):
            m[:, idx] = np.asarray(plydata.elements[0][attr_name])

        w1_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("w1_")]
        w1_names = sorted(w1_names, key=lambda x: int(x.split('_')[-1]))
        w1 = np.zeros((xyz.shape[0], len(w1_names)))
        for idx, attr_name in enumerate(w1_names):
            w1[:, idx] = np.asarray(plydata.elements[0][attr_name])

        sigma_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("sigma")]
        sigma_names = sorted(sigma_names, key=lambda x: int(x.split('_')[-1]))
        sigma = np.zeros((xyz.shape[0], len(sigma_names)))
        for idx, attr_name in enumerate(sigma_names):
            sigma[:, idx] = np.asarray(plydata.elements[0][attr_name])

        time_func_names = [p.name for p in plydata.elements[1].properties if p.name.startswith("time")]
        time_func_names = sorted(time_func_names, key=lambda x: int(x.split('_')[-1]))
        time_func = []
        for idx, attr_name in enumerate(time_func_names):
            time_func.append(plydata.elements[1][attr_name])
        time_func = np.array(time_func).flatten().reshape([-1, 1])

        if self.use_dff:
            self.time_func = nn.Parameter(torch.tensor(time_func, dtype=torch.float, device="cuda").requires_grad_(True))
        else:
            self.time_func = torch.from_numpy(time_func).to("cuda")

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(
            torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True)
        )
        self._features_rest = nn.Parameter(
            torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True)
        )
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self.m = nn.Parameter(torch.tensor(m, dtype=torch.float, device="cuda").requires_grad_(True))
        self.sigma = nn.Parameter(torch.tensor(sigma, dtype=torch.float, device="cuda").unsqueeze(-1).requires_grad_(True))
        self._w1 = nn.Parameter(torch.tensor(w1, dtype=torch.float, device="cuda").requires_grad_(True))

        self.polynomial_degree = len(w1_names) // 2
        self.active_sh_degree = self.max_sh_degree

        self._opacity_seg = torch.empty(0, device="cuda")
        self._features_dc_seg = torch.empty(0, device="cuda")
        self._features_rest_seg = torch.empty(0, device="cuda")

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                # this utility is only used for single-tensor groups
                assert len(group["params"]) == 1
                stored_state = self.optimizer.state.get(group['params'][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = torch.zeros_like(tensor)
                    stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                    del self.optimizer.state[group['params'][0]]
                    group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                    self.optimizer.state[group['params'][0]] = stored_state
                else:
                    group["params"][0] = nn.Parameter(tensor.requires_grad_(True))

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        """
        Prune ALL param groups that represent per-Gaussian tensors (including seg head groups if present).
        Supports both single-tensor groups and multi-tensor groups (rare; e.g. a custom "seg_head" group).
        """
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == 'time_func':
                continue

            params = group["params"]
            if len(params) == 1:
                p0 = params[0]
                stored_state = self.optimizer.state.get(p0, None)
                if stored_state is not None:
                    stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                    stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                    del self.optimizer.state[p0]
                    new_p = nn.Parameter((p0[mask].requires_grad_(True)))
                    group["params"][0] = new_p
                    self.optimizer.state[new_p] = stored_state
                    optimizable_tensors[group["name"]] = new_p
                else:
                    new_p = nn.Parameter(p0[mask].requires_grad_(True))
                    group["params"][0] = new_p
                    optimizable_tensors[group["name"]] = new_p
            else:
                # Multi-tensor group: prune each tensor
                new_params = []
                for p in list(params):
                    stored_state = self.optimizer.state.get(p, None)
                    if stored_state is not None:
                        stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                        stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]
                        del self.optimizer.state[p]
                        new_p = nn.Parameter(p[mask].requires_grad_(True))
                        self.optimizer.state[new_p] = stored_state
                        new_params.append(new_p)
                    else:
                        new_params.append(nn.Parameter(p[mask].requires_grad_(True)))
                group["params"] = new_params
                optimizable_tensors[group["name"]] = new_params

        return optimizable_tensors

# 3) Update prune_points(): assign seg tensors when present

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self.m = optimizable_tensors["m"]
        self.sigma = optimizable_tensors["sigma"]
        self._w1 = optimizable_tensors["w1"]


        if "f_dc_seg" in optimizable_tensors:
            self._features_dc_seg = optimizable_tensors["f_dc_seg"]
        if "f_rest_seg" in optimizable_tensors:
            self._features_rest_seg = optimizable_tensors["f_rest_seg"]
        if "opacity_seg" in optimizable_tensors:
            self._opacity_seg = optimizable_tensors["opacity_seg"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.m_gradient_accum = self.m_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.m_denom = self.m_denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]


    def cat_tensors_to_optimizer(self, tensors_dict):
        """
        Concatenate new per-Gaussian tensors to each optimizer parameter group.
        Supports single-tensor groups and multi-tensor groups.
        For multi-tensor groups, tensors_dict[group_name] must be a list/tuple with same length.
        """
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == 'time_func':
                continue

            gname = group["name"]
            if gname not in tensors_dict:
                raise KeyError(f"Missing tensors_dict entry for optimizer group '{gname}'")

            params = group["params"]
            ext = tensors_dict[gname]

            if len(params) == 1:
                assert isinstance(ext, torch.Tensor), f"Expected tensor for group '{gname}'"
                p0 = params[0]
                stored_state = self.optimizer.state.get(p0, None)
                if stored_state is not None:
                    stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(ext)), dim=0)
                    stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(ext)), dim=0)

                    del self.optimizer.state[p0]
                    new_p = nn.Parameter(torch.cat((p0, ext), dim=0).requires_grad_(True))
                    group["params"][0] = new_p
                    self.optimizer.state[new_p] = stored_state
                    optimizable_tensors[gname] = new_p
                else:
                    new_p = nn.Parameter(torch.cat((p0, ext), dim=0).requires_grad_(True))
                    group["params"][0] = new_p
                    optimizable_tensors[gname] = new_p
            else:
                assert isinstance(ext, (list, tuple)) and len(ext) == len(params), \
                    f"Expected list of {len(params)} tensors for group '{gname}'"
                new_params = []
                for p, e in zip(list(params), ext):
                    stored_state = self.optimizer.state.get(p, None)
                    if stored_state is not None:
                        stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(e)), dim=0)
                        stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(e)), dim=0)

                        del self.optimizer.state[p]
                        new_p = nn.Parameter(torch.cat((p, e), dim=0).requires_grad_(True))
                        self.optimizer.state[new_p] = stored_state
                        new_params.append(new_p)
                    else:
                        new_params.append(nn.Parameter(torch.cat((p, e), dim=0).requires_grad_(True)))
                group["params"] = new_params
                optimizable_tensors[gname] = new_params

        return optimizable_tensors

# 4) Update densification_postfix(): concatenate seg tensors too (if present)

    def densification_postfix(
        self,
        new_xyz, new_features_dc, new_features_rest, new_opacities,
        new_scaling, new_rotation, new_m, new_sigma, new_w1,
        new_features_dc_seg=None, new_features_rest_seg=None, new_opacities_seg=None
    ):
        d = {
            "xyz": new_xyz,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "opacity": new_opacities,
            "scaling": new_scaling,
            "rotation": new_rotation,
            "m": new_m,
            "sigma": new_sigma,
            "w1": new_w1,
        }


        if self.has_seg_head():

            if new_features_dc_seg is None: new_features_dc_seg = new_features_dc
            if new_features_rest_seg is None: new_features_rest_seg = new_features_rest
            if new_opacities_seg is None: new_opacities_seg = new_opacities

            d["f_dc_seg"] = new_features_dc_seg
            d["f_rest_seg"] = new_features_rest_seg
            d["opacity_seg"] = new_opacities_seg

        optimizable_tensors = self.cat_tensors_to_optimizer(d)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self.m = optimizable_tensors["m"]
        self.sigma = optimizable_tensors["sigma"]
        self._w1 = optimizable_tensors["w1"]


        if "f_dc_seg" in optimizable_tensors:
            self._features_dc_seg = optimizable_tensors["f_dc_seg"]
        if "f_rest_seg" in optimizable_tensors:
            self._features_rest_seg = optimizable_tensors["f_rest_seg"]
        if "opacity_seg" in optimizable_tensors:
            self._opacity_seg = optimizable_tensors["opacity_seg"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.m_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.m_denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grads_m, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]

        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values > self.percent_dense * scene_extent
        )

        padded_grad_m_tmp = torch.zeros((n_init_points), device="cuda")
        padded_grad_m_tmp[:grads_m.shape[0]] = torch.sign(grads_m).squeeze()

        padded_grad_m = torch.zeros((n_init_points), device="cuda")
        padded_grad_m[:grads_m.shape[0]] = grads_m.abs().squeeze()
        selected_pts_mask_m = torch.where(padded_grad_m >= grad_threshold, True, False)

        selected_pts_mask_scale = self.get_sigma.squeeze(-1).squeeze(-1) >= 1.0
        selected_pts_mask_m = torch.logical_and(selected_pts_mask_m, selected_pts_mask_scale)

        selected_pts_mask = torch.logical_or(selected_pts_mask, selected_pts_mask_m)

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)

        new_xyz = (
            torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1)
            + self.get_xyz[selected_pts_mask].repeat(N, 1)
        )
        new_scaling = self.scaling_inverse_activation(
            self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N)
        )
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)

        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)

        new_m = self.m[selected_pts_mask].repeat(N, 1)
        new_sigma = self.sigma[selected_pts_mask].repeat(N, 1, 1)
        new_w1 = self._w1[selected_pts_mask].repeat(N, 1)


        new_features_dc_seg = None
        new_features_rest_seg = None
        new_opacity_seg = None
        if self.has_seg_head():
            new_features_dc_seg = self._features_dc_seg[selected_pts_mask].repeat(N, 1, 1)
            new_features_rest_seg = self._features_rest_seg[selected_pts_mask].repeat(N, 1, 1)
            new_opacity_seg = self._opacity_seg[selected_pts_mask].repeat(N, 1)

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_scaling,
            new_rotation,
            new_m,
            new_sigma,
            new_w1,
            new_features_dc_seg=new_features_dc_seg,
            new_features_rest_seg=new_features_rest_seg,
            new_opacities_seg=new_opacity_seg,
        )

        prune_filter = torch.cat(
            (selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool))
        )
        self.prune_points(prune_filter)


    def densify_and_clone(self, grads, grads_m, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values <= self.percent_dense * scene_extent
        )

        selected_pts_mask_m = torch.where(grads_m.abs().squeeze(1) >= grad_threshold, True, False)
        selected_pts_mask_scalem = self.get_sigma.squeeze(-1).squeeze(-1) <= 1.0
        selected_pts_mask_m = torch.logical_and(selected_pts_mask_m, selected_pts_mask_scalem)

        selected_pts_mask = torch.logical_or(selected_pts_mask, selected_pts_mask_m)

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_m = self.m[selected_pts_mask]
        new_sigma = self.sigma[selected_pts_mask]
        new_w1 = self._w1[selected_pts_mask]


        new_features_dc_seg = None
        new_features_rest_seg = None
        new_opacities_seg = None
        if self.has_seg_head():
            new_features_dc_seg = self._features_dc_seg[selected_pts_mask]
            new_features_rest_seg = self._features_rest_seg[selected_pts_mask]
            new_opacities_seg = self._opacity_seg[selected_pts_mask]

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacities,
            new_scaling,
            new_rotation,
            new_m,
            new_sigma,
            new_w1,
            new_features_dc_seg=new_features_dc_seg,
            new_features_rest_seg=new_features_rest_seg,
            new_opacities_seg=new_opacities_seg,
        )


    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        grads_m = self.m_gradient_accum / self.m_denom
        grads_m[grads_m.isnan()] = 0.0

        self.densify_and_clone(grads, grads_m, max_grad, extent)
        self.densify_and_split(grads, grads_m, max_grad, extent)

        prune_mask_img = (self.get_opacity_head("img") < min_opacity).squeeze()

        if self.has_seg_head():
            prune_mask_seg = (self.get_opacity_head("seg") < min_opacity).squeeze()
            prune_mask = prune_mask_img & prune_mask_seg
        else:
            prune_mask = prune_mask_img

        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * 1.1
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)

        self.prune_points(prune_mask)
        torch.cuda.empty_cache()


    def add_densification_stats(self, xyscreen, update_filter_batch):
        grads_scale = [d.grad[update_filter_batch, :2].detach() * 1 for d in xyscreen]
        grads_norm = [d.norm(dim=-1, keepdim=True) for d in grads_scale]
        grads_max = torch.cat(grads_norm, dim=-1).max(dim=-1)[0]
        grads = grads_max[:, None]

        self.xyz_gradient_accum[update_filter_batch] += grads
        self.denom[update_filter_batch] += 1
        self.m_gradient_accum[update_filter_batch] += self.m.grad[update_filter_batch, :1].abs()
        self.m_denom[update_filter_batch] += 1


