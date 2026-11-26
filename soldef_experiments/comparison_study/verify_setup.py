"""Instantiate both class-count adapters and verify pretrained coverage without GPU computation."""
import argparse,json,subprocess,sys
import torch
from common import ROOT,set_seed,write_json,digest
from comparison_study.train import load_config,TrainingLoss
from modern_study.model import create_model

def main():
 p=argparse.ArgumentParser();p.add_argument('--config');a=p.parse_args()
 if not a.config:
  paths=sorted((ROOT/'configs/comparison_study').glob('comparison_*.json'))
  if len(paths)!=12:raise ValueError('Expected 12 prepared comparison configs')
  for path in paths:subprocess.run([sys.executable,'-m','comparison_study.verify_setup','--config',str(path)],cwd=ROOT,check=True)
  return
 cfg=load_config(a.config)
 torch.set_num_threads(2);set_seed(42);model=create_model(cfg);assert all(p.device.type=='cpu' for p in model.parameters())
 c=len(cfg['classes']);name=cfg['model']['name'];tag=cfg['task']+'_'+name;checks=['config validation','CPU construction','pretrained key and tensor-shape validation']
 if name not in ('segman_s','mambavision_t'):
  x=torch.randn(2,3,64,64);y=torch.randint(0,c,(2,64,64));model.train();out=model(x)
  logits=out['out'] if isinstance(out,dict) else out;assert logits.shape==(2,c,64,64)
  loss=TrainingLoss(cfg)(out,y);assert torch.isfinite(loss);loss.backward()
  grads=[p.grad for p in model.parameters() if p.grad is not None];assert grads and all(torch.isfinite(g).all() for g in grads)
  checks+=['CPU 64x64 multi-class forward/backward']
 else:
  assert model.head.num_classes==c
  if hasattr(model,'auxiliary_head'):assert model.auxiliary_head.num_classes==c
  checks+=['main/auxiliary class counts'];loss=None
 result={'passed':True,'task':cfg['task'],'model':name,'classes':cfg['classes'],'parameters':sum(p.numel() for p in model.parameters()),
 'checks':checks,'full_size_GPU_preflight':'queued after predecessor completion; not yet run',
 'config_sha256':digest(a.config),'initialization':getattr(model,'initialization_provenance',{}),'test_split_accessed':False}
 write_json(ROOT/'reports/comparison_study'/f'setup_{tag}.json',result);print('SETUP PASS',tag,flush=True)
if __name__=='__main__':main()
