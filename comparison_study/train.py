"""Fixed-budget SolDef author-model comparisons; validation-only model selection."""
import argparse,ast,itertools,json,shutil,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from common import ROOT,digest,set_seed,write_json
from experiment import architecture_info,environment,epoch_pass
from innovation_study.stable_loss import StablePaperLoss
from metrics import ConfusionMatrix
from paper_protocol.data import audit,make_loader
from soldef_study.train import learning_rate,save,check_sources,runtime_hashes as focus_hashes
from modern_study.model import create_model

def runtime_hashes():
 result=focus_hashes()
 for p in (ROOT/'comparison_study').glob('*.py'):result[str(p.relative_to(ROOT))]=digest(p)
 paths=list((ROOT/'modern_study').glob('*.py'))+list((ROOT/'modern_study/vendor').rglob('*.py'))
 for p in paths:
  if '.git' not in p.parts:result[str(p.relative_to(ROOT))]=digest(p)
 return result

def load_config(path):
 c=json.loads(Path(path).read_text());t=c['training']
 assert c['protocol_id']=='two_task_comparison_v1' and c['dataset'] in ('soldef','soldef_task1')
 assert (c['image_size'],t['epochs'],t['batch_size'],t['lr'])==(512,150,2,1e-4)
 assert not t['mixed_precision'] and t['selection']=='min_val_loss' and t['scheduler']=='cosine'
 assert c['model']['name'] in ['emcad','segman_s','mambavision_t','segformer_b2','unet2015','deeplabv3_r101_paper']
 expected={'task1':('soldef_task1',['background','good','no_good']), 'task2':('soldef',['background','good','exc_solder','poor_solder','spike'])}
 assert (c['dataset'],c['classes'])==expected[c['task']]
 assert c['split_protocol']=='source_group'
 return c

class TrainingLoss(nn.Module):
 def __init__(self,cfg):
  super().__init__();self.name=cfg['model']['name'];self.common=StablePaperLoss(**cfg['loss'])
  if self.name=='emcad':
   # Execute ONLY the reviewed, self-contained author DiceLoss class, not utility imports.
   path=ROOT/'modern_study/vendor/EMCAD/utils/utils.py'
   tree=ast.parse(path.read_text());node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='DiceLoss')
   ns={'torch':torch,'nn':nn};exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),ns)
   self.dice=ns['DiceLoss'](len(cfg['classes']))
 def forward(self,output,target):
  main=output['out'] if isinstance(output,dict) else output
  if self.name=='emcad':
   heads=output['multi'];assert len(heads)==4
   loss=main.new_zeros(())
   for k in range(1,5):
    for subset in itertools.combinations(range(4),k):
     logits=sum(heads[i] for i in subset)
     loss=loss+.3*F.cross_entropy(logits,target)+.7*self.dice(logits,target,softmax=True)
   return loss
  if self.name in ('segman_s','segformer_b2','mambavision_t'):
   loss=F.cross_entropy(main,target)
   if self.name=='mambavision_t':loss=loss+.4*F.cross_entropy(output['aux'],target)
   return loss
  return self.common(main,target)

def training_pass(model,loader,criterion,optimizer,classes,device):
 model.train();matrix=ConfusionMatrix(len(classes));total=0.;samples=0
 for x,y,_ in loader:
  x,y=x.to(device),y.to(device);optimizer.zero_grad(set_to_none=True)
  output=model(x);loss=criterion(output,y)
  if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
  loss.backward();optimizer.step();logits=output['out'] if isinstance(output,dict) else output
  matrix.update(logits.detach(),y);total+=loss.item()*len(x);samples+=len(x)
 return {'loss':total/samples,'samples':samples,**matrix.compute(classes)}

def train(cfg,seed,device,resume=False,stop_after=None):
 t=cfg['training'];run=ROOT/'comparison_runs'/cfg['experiment']/f'seed_{seed}'
 data=audit(ROOT/cfg['manifest']);assert data['manifest_sha256']==cfg['manifest_sha256']
 saved=None
 if resume:
  saved=torch.load(run/'last.pt',map_location='cpu',weights_only=True)
  assert saved['config']==cfg and saved['seed']==seed;check_sources(saved['source_sha256'])
 else:
  run.mkdir(parents=True,exist_ok=False);write_json(run/'config.json',cfg);shutil.copy2(ROOT/cfg['manifest'],run/'data_manifest.json')
 status={'status':'running','seed':seed,'test_split_accessed':False,'protocol_id':cfg['protocol_id']}
 write_json(run/'status.json',status)
 try:
  set_seed(seed);torch.set_num_threads(t['cpu_threads']);model=create_model(cfg,False if resume else None).to(device)
  optimizer=torch.optim.AdamW(model.parameters(),lr=t['lr'],weight_decay=t['weight_decay'])
  if saved:
   model.load_state_dict(saved['model_state_dict'],strict=True);optimizer.load_state_dict(saved['optimizer_state_dict'])
   history,best,hashes=saved['history'],saved['best'],saved['source_sha256'];start=saved['epoch'];elapsed_before=saved['elapsed_seconds'];del saved
  else:
   hashes=runtime_hashes();history=[];best={'loss':float('inf'),'epoch':0};start=0;elapsed_before=0.
   for name in hashes:
    target=run/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
   write_json(run/'architecture.json',architecture_info(model))
   write_json(run/'provenance.json',{'source_sha256':hashes,'data_audit':data,'environment':environment(),
              'initialization':getattr(model,'initialization_provenance',{}),'training_loss':cfg['training_loss']})
  criterion=TrainingLoss(cfg);val_criterion=StablePaperLoss(**cfg['loss']);began=time.perf_counter();limit=min(stop_after or t['epochs'],t['epochs'])
  for epoch in range(start,limit):
   set_seed(seed+epoch);lr=learning_rate(t,epoch)
   for g in optimizer.param_groups:g['lr']=lr
   training=training_pass(model,make_loader(cfg,'train',seed+epoch),criterion,optimizer,cfg['classes'],device)
   val,_=epoch_pass(model,make_loader(cfg,'val',seed),val_criterion,device,classes=cfg['classes'])
   elapsed=elapsed_before+time.perf_counter()-began
   history.append({'epoch':epoch+1,'lr':lr,'train':training,'val':val,'elapsed_seconds':elapsed})
   checkpoint={'config':cfg,'seed':seed,'epoch':epoch+1,'source_sha256':hashes,'manifest_sha256':data['manifest_sha256'],'model_state_dict':model.state_dict()}
   if val['loss']<best['loss']:
    best={'loss':val['loss'],'epoch':epoch+1};save({**checkpoint,'val':val},run/'best.pt')
   save({**checkpoint,'optimizer_state_dict':optimizer.state_dict(),'history':history,'best':best,'elapsed_seconds':elapsed},run/'last.pt')
   write_json(run/'history.json',history);status.update(epoch=epoch+1,best=best,elapsed_seconds=elapsed);write_json(run/'status.json',status)
   print(f"{cfg['experiment']} seed={seed} epoch={epoch+1}/150 val={val['loss']:.6f} mIoU={val['mIoU']:.6f} elapsed={elapsed:.1f}s",flush=True)
  status.update(status='completed' if limit==150 else 'paused');write_json(run/'status.json',status)
 except BaseException as e:status.update(status='failed',error=repr(e));write_json(run/'status.json',status);raise

def evaluate(run,device):
 run=Path(run);status=json.loads((run/'status.json').read_text());assert status['status']=='completed' and status['epoch']==150
 ckpt=torch.load(run/'best.pt',map_location='cpu',weights_only=True);cfg=ckpt['config'];check_sources(ckpt['source_sha256'])
 assert audit(ROOT/cfg['manifest'])['manifest_sha256']==ckpt['manifest_sha256']
 set_seed(ckpt['seed']);torch.set_num_threads(cfg['training']['cpu_threads']);model=create_model(cfg,False).to(device)
 model.load_state_dict(ckpt['model_state_dict'],strict=True)
 value,cm=epoch_pass(model,make_loader(cfg,'val',ckpt['seed']),StablePaperLoss(**cfg['loss']),device,classes=cfg['classes'])
 for k in ['loss','mIoU','mDice','mPA']:
  if not np.isclose(value[k],ckpt['val'][k],atol=1e-7,rtol=1e-6):raise ValueError('Reload mismatch '+k)
 write_json(run/'validation_metrics.json',{'epoch':ckpt['epoch'],'split':'val','reloaded_checkpoint_verified':True,**value})
 np.savetxt(run/'validation_confusion.csv',cm,fmt='%d',delimiter=',')
 status.update(evaluation_completed=True,validation_reload_verified=True);write_json(run/'status.json',status)

def main():
 p=argparse.ArgumentParser();p.add_argument('action',choices=['train','evaluate']);p.add_argument('--config',required=True);p.add_argument('--seed',type=int,default=42);p.add_argument('--resume',action='store_true');p.add_argument('--stop-after',type=int);p.add_argument('--device',default='cuda');a=p.parse_args()
 cfg=load_config(a.config);run=ROOT/'comparison_runs'/cfg['experiment']/f'seed_{a.seed}'
 if a.action=='train':train(cfg,a.seed,torch.device(a.device),a.resume,a.stop_after)
 else:evaluate(run,torch.device(a.device))
if __name__=='__main__':main()
