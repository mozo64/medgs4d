import open3d as o3d
import numpy as np
import trimesh

def scale_mesh_to_match(source_mesh, target_mesh):
    source_size = source_mesh.extents
    target_size = target_mesh.extents

    scale_factors = target_size / source_size
    uniform_scale = np.mean(scale_factors)

    scale_matrix = np.eye(4)
    scale_matrix[:3, :3] *= uniform_scale

    source_mesh.apply_transform(scale_matrix)
    return source_mesh

def trimesh_to_open3d(mesh):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(mesh.vertices)
    return pcd

def icp_align(source_mesh, target_mesh, threshold=20.0):
    src_vertices = np.copy(source_mesh.vertices)
    src_faces = np.copy(source_mesh.faces)
    src_normals = (
        np.copy(source_mesh.vertex_normals)
        if source_mesh.vertex_normals is not None and len(source_mesh.vertex_normals) == len(source_mesh.vertices)
        else None
    )

    def trimesh_to_open3d(mesh_vertices):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(mesh_vertices)
        return pcd

    source_pcd = trimesh_to_open3d(src_vertices)
    target_pcd = trimesh_to_open3d(target_mesh.vertices)

    source_pcd.estimate_normals()
    target_pcd.estimate_normals()

    trans_init = np.eye(4)
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source_pcd, target_pcd, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )

    # Transform copied vertices and normals
    transformed_vertices = trimesh.transformations.transform_points(src_vertices, reg_p2p.transformation)

    if src_normals is not None:
        R = reg_p2p.transformation[:3, :3]
        transformed_normals = src_normals @ R.T
    else:
        transformed_normals = None

    result_mesh = trimesh.Trimesh(
        vertices=transformed_vertices,
        faces=src_faces,
        vertex_normals=transformed_normals,
        process=False
    )

    return result_mesh, reg_p2p.transformation

def rotation_matrix_x(angle_degrees):
    angle = np.radians(angle_degrees)
    R = np.eye(4)
    R[1, 1] = np.cos(angle)
    R[1, 2] = -np.sin(angle)
    R[2, 1] = np.sin(angle)
    R[2, 2] = np.cos(angle)
    return R

def rotation_matrix_y(angle_degrees):
    angle = np.radians(angle_degrees)
    R = np.eye(4)
    R[0, 0] = np.cos(angle)
    R[0, 2] = np.sin(angle)
    R[2, 0] = -np.sin(angle)
    R[2, 2] = np.cos(angle)
    return R

def rotation_matrix_z(angle_degrees):
    angle = np.radians(angle_degrees)
    R = np.eye(4)
    R[0, 0] = np.cos(angle)
    R[0, 1] = -np.sin(angle)
    R[1, 0] = np.sin(angle)
    R[1, 1] = np.cos(angle)
    return R

def rotate_mesh(mesh, x,y,z):
    Rx = rotation_matrix_x(x)
    Ry = rotation_matrix_y(y)
    Rz = rotation_matrix_z(z)
    combined_rotation = Rx @ Ry @ Rz

    res_mesh = mesh.copy()
    res_mesh.apply_transform(combined_rotation)
    return res_mesh

def prepare_mesh(pred_mesh_, gt_mesh, x, y, z, threshold=20.0):
    pred_mesh = rotate_mesh(pred_mesh_, x, y, z)
    pred_mesh, _ = icp_align(pred_mesh, gt_mesh, threshold)
    return pred_mesh


