"""CPU integration checks with a differentiable substitute for the CUDA renderer."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np
import pandas as pd
import pytest
import torch

from arguments.temporal import (CanonicalConfig, DeformationConfig, MedGS4DConfig,
    TrainingConfig, SplitConfig, load_medgs4d_config)
from scene.volume_data import StudyManifest
from training.methods import METHODS, create_field, training_step_for
from utils.run_utils import build_run_paths
from training import temporal as tr


def fixtures(tmp_path, method):
    torch.manual_seed(101)
    xyz, logits = torch.randn(17, 3), torch.randn(17, 1)
    def render(camera, view, pipeline, background, **kwargs):
        value = (view.get_xyz[:, [0, 2]].mean() + view.get_m.mean()).sigmoid()
        return {"render": value.expand(3, 8, 8)}
    canonical = SimpleNamespace(xyz=xyz, xz=xyz[:, [0, 2]], m_logits=logits,
        m=logits.sigmoid(), gaussians=SimpleNamespace(), cameras=[None],
        pipeline=None, background=None,
        runtime=SimpleNamespace(render=render,
            l1_loss=lambda a,b: (a-b).abs().mean(),
            ssim=lambda a,b: 1-(a-b).square().mean()))
    paths={}
    for phase, value in [(0.0, 0.3), (50.0, 0.7)]:
        p=tmp_path / f'phase{phase}.npy'
        np.save(p,np.full((1,8,8),value,dtype=np.float32)); paths[phase]=str(p)
    manifest=tmp_path/'slices.csv'
    pd.DataFrame({'PhasePercent':[0,50], 'SliceIndex':[0,0], 'SliceCoordinate':[0,0]}).to_csv(manifest,index=False)
    study=StudyManifest('test','test','test',(0.0,50.0),1,(1,8,8),(0,1),(0,0,0),paths,paths,str(manifest),str(manifest))
    kw=dict(iterations=6, log_every=2, checkpoint_every=2, validate_every=0,
        validation_samples=0, smoothness_gaussians=17,
        pairwise_hidden_dim=8, pairwise_hidden_layers=1,
        pairwise_cycle_warmup_iterations=1,pairwise_cycle_ramp_iterations=1)
    if method=='linear_pseudo':kw['pseudo_weight']=0.1
    if method=='cycle_v2':kw.update(pairwise_cycle_weight=.05,pairwise_agreement_weight=.01,pairwise_transport_weight=.001)
    options=dict(endpoint_warmup=1,transport_pretrain=1,joint_ramp=1,
        endpoint_image_weight=.05,alignment_weight=.01,roundtrip_weight=.01,path_weight=.005,
        correction_weight=1e-3,curvature_weight=1e-5,curvature_step=.025,geometry_gaussians=13)
    config=MedGS4DConfig('test',method,str(tmp_path),str(tmp_path),0.0,
        SplitConfig(),DeformationConfig(hidden_dim=8,hidden_layers=1,chunk_size=9),
        TrainingConfig(**kw),method=method,method_options=options)
    return canonical,study,config


@pytest.mark.parametrize('method',METHODS)
def test_train_save_reload_and_resume(tmp_path,method):
    canonical,study,config=fixtures(tmp_path,method)
    paths=build_run_paths(tmp_path/'runs','test',method)
    paths.root.mkdir(parents=True)
    original_class, original_step = tr.DeformationField, tr.train_step
    tr.train_medgs4d(study,canonical,paths,config,final_evaluation=False)
    assert tr.DeformationField is original_class and tr.train_step is original_step
    restored_config=load_medgs4d_config(paths.config)
    field=create_field(canonical,restored_config)
    checkpoint=paths.checkpoints/'deformation_iter_000006.pth'
    assert tr.load_checkpoint(checkpoint,field,device='cpu',config=config)==6
    prediction=field.predict_relative_deformation(.2,use_checkpointing=False).detach()
    assert torch.isfinite(prediction).all()
    assert torch.count_nonzero(prediction)>0
    payload=torch.load(checkpoint,weights_only=False)
    assert payload['config']['method']==method
    assert ('pairwise_cycle_state_dict' in payload)==(method in {'cycle_v2','consistent_cycle','trajectory_cycle'})
    history=pd.read_csv(paths.training_history)
    assert np.isfinite(history.TotalLoss).all()
    if method in {'consistent_cycle','trajectory_cycle'}:
        assert history.TransportEndpointImageLoss.iloc[-1]>0
    # Resume from an earlier checkpoint and reproduce the final primary weights exactly.
    tr.train_medgs4d(study,canonical,paths,config,resume=True,
        resume_checkpoint=paths.checkpoints/'deformation_iter_000004.pth',final_evaluation=False)
    again=create_field(canonical,config)
    tr.load_checkpoint(checkpoint,again,device='cpu',config=config)
    assert torch.equal(prediction,again.predict_relative_deformation(.2,use_checkpointing=False).detach())
    bad=replace(config,method='baseline' if method!='baseline' else 'linear_pseudo')
    with pytest.raises(ValueError,match='method'):
        tr.load_checkpoint(checkpoint,field,device='cpu',config=bad)


def test_worker_selftest():
    from benchmark.worker import selftest
    selftest()


def test_canonical_invocation_uses_shared_script_and_explicit_seed(tmp_path,monkeypatch):
    from scene import canonical
    calls=[]
    monkeypatch.setattr(canonical.subprocess,'run',lambda command,**kw:calls.append((command,kw)))
    config=CanonicalConfig('test','test',0.0,iterations=20,seed=42)
    canonical.run_medgs_training(tmp_path/'data',tmp_path/'model',config,
        training_history=tmp_path/'history.csv',log_every=10)
    command,kw=calls[0]
    assert Path(command[1])==Path(canonical.__file__).resolve().parents[1]/'train.py'
    assert command[command.index('--seed')+1]=='42'
    assert command[command.index('--history_log_every')+1]=='10'
    assert not list(tmp_path.glob('**/*.py'))


def test_metrics_and_notebook_contract():
    import ast, nbformat
    from benchmark.metrics import ssim3d,ncc3d
    x=torch.rand(1,1,17,17,17)
    assert abs(float(ssim3d(x,x))-1)<1e-6
    assert float(ncc3d(x,x))>.999
    path=Path(__file__).resolve().parents[1]/'notebooks/09_MedGS4D_publication_benchmark.ipynb'
    notebook=nbformat.read(path,as_version=4);nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type=='code':
            ast.parse(cell.source)
            assert not cell.outputs
    source='\n'.join(c.source for c in notebook.cells)
    for obsolete in ['SOURCE_ARCHIVES','METHOD_SOURCE_PARTS','WORKER_SOURCE_PARTS','install_training_variant']:
        assert obsolete not in source
