import copy,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pytest,torch
from common import ROOT,digest,set_seed
from soldef_study.model import VARIANTS,create_model,assert_architecture

@pytest.mark.parametrize('variant',VARIANTS)
def test_models_share_all_common_initialization_and_have_valid_gradients(variant):
    torch.set_num_threads(2)
    cfg=json.loads((ROOT/f'configs/two_task/twotask_task2_{variant}.json').read_text());cfg['image_size']=64
    basecfg=copy.deepcopy(cfg);basecfg['study_variant']='none'
    set_seed(42);base=create_model(basecfg,pretrained=False,seed=42).eval()
    set_seed(42);model=create_model(cfg,pretrained=False,seed=42).eval();assert_architecture(cfg,model)
    assert model.initialization_provenance['shared_state_sha256']==base.initialization_provenance['shared_state_sha256']
    for key,value in base.state_dict().items():torch.testing.assert_close(model.state_dict()[key],value,rtol=0,atol=0)
    x=torch.randn(2,3,64,64);pred=model(x);assert pred.shape==(2,5,64,64)
    if variant.startswith('residual_'):
        with torch.no_grad():reference=base(x)
        torch.testing.assert_close(pred,reference,rtol=0,atol=0)
    pred.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    if variant.startswith('residual_'):
        assert all(b.attention.alpha.grad.abs()>0 for b in [*model.decoder,model.final_fusion])


def test_resume_schedule_selection_and_validation_boundary(tmp_path,monkeypatch):
    import soldef_study.train as rt
    (tmp_path/'manifest.json').write_text('{}')
    monkeypatch.setattr(rt,'ROOT',tmp_path);monkeypatch.setattr(rt,'audit',lambda path:{'manifest_sha256':'fixed'})
    monkeypatch.setattr(rt,'runtime_hashes',lambda:{})
    monkeypatch.setattr(rt,'environment',lambda:{'test':'cpu'})
    monkeypatch.setattr(rt,'create_model',lambda cfg,pretrained=None,seed=42:torch.nn.Conv2d(3,2,1))
    monkeypatch.setattr(rt,'assert_architecture',lambda cfg,model:None);monkeypatch.setattr(rt,'validate_config',lambda cfg:None)
    seen=[]
    def loader(cfg,split,seed,batch_size=None):
        assert split in {'train','val'};seen.append(split);g=torch.Generator().manual_seed(seed)
        return [(torch.randn(2,3,8,8,generator=g),torch.randint(0,2,(2,8,8),generator=g),['a','b'])]
    monkeypatch.setattr(rt,'make_loader',loader)
    cfg={'experiment':'continuous','manifest':'manifest.json','manifest_sha256':'fixed','protocol_id':'toy',
         'classes':['background','fg'],'model':{},'loss':{},
         'training':{'epochs':150,'lr':1e-4,'min_lr':1e-6,'weight_decay':1e-5,'cpu_threads':2}}
    a=rt.train(cfg,42,torch.device('cpu'),stop_after=3);cfg=copy.deepcopy(cfg);cfg['experiment']='resumed'
    b=rt.train(cfg,42,torch.device('cpu'),stop_after=1);rt.train(cfg,42,torch.device('cpu'),resume=True,stop_after=3)
    aa=torch.load(a/'last.pt',weights_only=True);bb=torch.load(b/'last.pt',weights_only=True)
    for k in aa['model_state_dict']:torch.testing.assert_close(aa['model_state_dict'][k],bb['model_state_dict'][k],rtol=0,atol=0)
    assert [h['val']['loss'] for h in aa['history']]==[h['val']['loss'] for h in bb['history']]
    with pytest.raises(ValueError,match='150'):rt.evaluate(b,torch.device('cpu'))
    rt.train(cfg,42,torch.device('cpu'),resume=True);rt.evaluate(b,torch.device('cpu'))
    h=json.loads((b/'history.json').read_text());s=json.loads((b/'status.json').read_text())
    assert h[0]['lr']==pytest.approx(1e-4) and h[-1]['lr']==pytest.approx(1e-6)
    assert all(a['lr']>=b['lr'] for a,b in zip(h,h[1:]))
    assert s['best']['epoch']==min(h,key=lambda row:row['val']['loss'])['epoch']
    assert s['validation_reload_verified'] and s['evaluation_completed'] and not s['test_split_accessed']
    assert set(seen)=={'train','val'} and not list(b.glob('test*'))


def test_fixed_two_task_matrix_and_reference_config_hashes():
    from soldef_study.train import validate_config
    plan=json.loads((ROOT/'configs/two_task/plan.json').read_text())
    assert len(plan['jobs'])==90
    assert len({(e['config'],e['seed']) for e in plan['jobs']})==90
    for e in plan['jobs']:
        assert digest(ROOT/e['config'])==e['config_sha256']
        cfg=json.loads((ROOT/e['config']).read_text());validate_config(cfg)
        assert cfg['split_protocol']=='source_group'
        assert cfg['model']['backbone']=='convnext_base'
    assert plan['seeds']==[42,43,44,45,46]


def test_cli_loads_cosine_config_and_dispatches_soldef(tmp_path,monkeypatch):
    import soldef_study.train as rt
    config=ROOT/'configs/two_task/twotask_task2_none.json'
    loaded=rt.load_config(config);assert loaded['training']['scheduler']=='cosine'
    called=[]
    monkeypatch.setattr(rt,'ROOT',tmp_path)
    monkeypatch.setattr(rt,'device_for',lambda value:torch.device('cpu'))
    monkeypatch.setattr(rt,'train',lambda cfg,seed,device,resume,stop_after:called.append((cfg,seed,resume,stop_after)))
    monkeypatch.setattr(sys,'argv',['soldef_study.train','train','--config',str(config),'--seed','42','--stop-after','1'])
    rt.main()
    assert len(called)==1 and called[0][0]['dataset']=='soldef' and called[0][3]==1
    bad=copy.deepcopy(loaded);bad['dataset']='visa';path=tmp_path/'invalid.json';path.write_text(json.dumps(bad))
    with pytest.raises(AssertionError):rt.load_config(path)
