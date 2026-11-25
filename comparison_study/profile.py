"""Profiles for both class counts; called serially after training, under queue locks."""
import argparse,json,subprocess,time
import numpy as np
import torch
from common import ROOT,set_seed,write_json,digest
from comparison_study.report import all_jobs

def main():
 p=argparse.ArgumentParser();p.add_argument('--task',required=True);p.add_argument('--model',required=True);a=p.parse_args()
 # Parent supervisor owns both training locks; reject direct concurrent profiling.
 state=json.loads((ROOT/'reports/comparison_study/queue_status.json').read_text())
 import os
 assert state['pid']==os.getppid() and state['phase']=='profiling','Only the exclusive supervisor may dispatch profiling'
 util=[]
 for _ in range(3):
  value=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu','--format=csv,noheader,nounits'],text=True)
  util.append(int(value.splitlines()[0]));time.sleep(1)
 if max(util)>5:raise SystemExit(75)
 job=next(j for j in all_jobs() if j['task']==a.task and j['model']==a.model);cfg=json.loads((ROOT/job['config']).read_text())
 torch.set_num_threads(2);set_seed(42);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 if job['kind']=='ablation':
  from soldef_study.model import create_model
  model=create_model(cfg,False,seed=42)
 else:
  from modern_study.model import create_model
  model=create_model(cfg,False)
 model=model.cuda().eval();x=torch.randn(1,3,512,512,device='cuda');times=[]
 with torch.inference_mode():
  for _ in range(30):model(x)
  torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
  for _ in range(200):
   torch.cuda.synchronize();start=time.perf_counter();model(x);torch.cuda.synchronize();times.append(time.perf_counter()-start)
  peak=torch.cuda.max_memory_allocated()/2**20
  with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],with_flops=True) as prof:model(x)
  torch.cuda.synchronize()
 counted={e.key:e.flops for e in prof.key_averages() if e.flops}
 value={'task':a.task,'model':a.model,'classes':cfg['classes'],'input_shape':[1,3,512,512],'parameters':sum(p.numel() for p in model.parameters()),
 'supported_operator_GFLOPs':sum(counted.values())/1e9,'counted_operator_FLOPs':counted,'complete_flops':False,
 'uncounted_examples':['normalization','activation','pooling','interpolation','custom selective scan','custom NATTEN','fused SDPA'],
 'flops_convention':'multiply-add=2; supported operators only','latency_valid':True,'fps':float(1/np.mean(times)),
 'latency_ms':float(1000*np.mean(times)),'latency_std_ms':float(1000*np.std(times,ddof=1)),
 'peak_allocated_MiB':peak,'device':torch.cuda.get_device_name(),'torch':str(torch.__version__),'cuda':torch.version.cuda,
 'dtype':'float32','tf32_matmul':False,'tf32_cudnn':False,'warmup':30,'iterations':200,'gpu_utilization_before':util,
 'scope':'Random-weight structural profiling; all parameters including auxiliary/inactive heads; eager forward-only excludes IO/preprocessing/transfers',
 'config_sha256':digest(ROOT/job['config']),'source_sha256':digest(__file__)}
 write_json(ROOT/'reports/comparison_study/profiles'/f'{a.task}_{a.model}.json',value)
 print('PROFILE',a.task,a.model,value['fps'],flush=True)
if __name__=='__main__':main()
