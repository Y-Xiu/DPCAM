"""Full-size real-train-batch gradient/serialization checks; no validation or test tuning."""
import argparse,gc,json,tempfile,time
from pathlib import Path
import torch
from common import ROOT,set_seed,write_json
from modern_study.model import create_model
from comparison_study.train import TrainingLoss,load_config
from paper_protocol.data import make_loader

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args()
 cfg=load_config(a.config);tag=cfg['task']+'_'+cfg['model']['name']
 torch.set_num_threads(2);set_seed(42);began=time.perf_counter()
 x,y,ids=next(iter(make_loader(cfg,'train',42,workers=0)));x,y=x.cuda(),y.cuda()
 model=create_model(cfg).cuda();model.train();optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['training']['lr'])
 torch.cuda.reset_peak_memory_stats();output=model(x);loss=TrainingLoss(cfg)(output,y)
 assert torch.isfinite(loss);loss.backward()
 gradients=[(n,p.grad) for n,p in model.named_parameters() if p.grad is not None]
 assert gradients and all(torch.isfinite(g).all() for _,g in gradients)
 assert any(g.abs().max()>0 for _,g in gradients)
 optimizer.step();optimizer.zero_grad(set_to_none=True);del optimizer,output
 model.eval()
 with torch.inference_mode():
  output=model(x);pred=output['out'] if isinstance(output,dict) else output
  assert pred.shape==(2,len(cfg['classes']),512,512) and torch.isfinite(pred).all()
  expected=pred.detach().cpu()
 path=ROOT/'reports/comparison_study'/f'{tag}_preflight_state.pt'
 torch.save(model.state_dict(),path);model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True),strict=True);path.unlink()
 with torch.inference_mode():
  output=model(x);pred=output['out'] if isinstance(output,dict) else output
  assert torch.allclose(expected,pred.cpu(),atol=1e-6,rtol=1e-5)
 value={'passed':True,'model':cfg['model']['name'],'task':cfg['task'],'input_shape':list(x.shape),'classes':cfg['classes'],'train_crop_ids':list(ids),
 'loss':loss.item(),'gradient_tensors':len(gradients),'parameters':sum(p.numel() for p in model.parameters()),
 'peak_allocated_MiB':torch.cuda.max_memory_allocated()/2**20,'elapsed_seconds':time.perf_counter()-began,
 'initialization':getattr(model,'initialization_provenance',{}),'test_split_accessed':False,'accuracy_result':False,
 'checks':['512 batch2 FP32 forward','finite nonzero backward gradients','AdamW step','strict weights-only reload','eval-output agreement']}
 write_json(ROOT/'reports/comparison_study'/f'preflight_{tag}.json',value);print('PASSED',json.dumps(value),flush=True)
if __name__=='__main__':main()
