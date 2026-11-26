"""Wait for Trae's 90 runs, then own both locks and dispatch 60 comparisons.
A failed/stopped predecessor never counts as completion. Restart is resumable.
"""
import argparse,contextlib,fcntl,importlib.metadata,json,math,os,shutil,subprocess,sys,sysconfig,time
from pathlib import Path
from common import ROOT,digest,write_json
from soldef_study.train import check_sources
from comparison_study.train import runtime_hashes
from comparison_study.report import all_jobs,generate,metric_values
FOLDER=ROOT/'reports/comparison_study'
PYTHON=Path(sys.executable)

def validated_run(root,run,job,epochs=150):
 """Check completed-run evidence without loading weights or accessing image data."""
 try:
  status=json.loads((run/'status.json').read_text())
  if not (status.get('status')=='completed' and status.get('epoch')==epochs and
          status.get('evaluation_completed') and status.get('validation_reload_verified')):
   return False,'run_not_validated'
  for name in ('best.pt','validation_metrics.json','config.json'):
   path=run/name
   if not path.is_file() or path.stat().st_size==0:return False,'artifact_missing_or_empty:'+name
  cfg=json.loads((run/'config.json').read_text())
  if not isinstance(cfg,dict):return False,'invalid_run_config'
  if 'config' in job:
   source=root/job['config']
   if cfg!=json.loads(source.read_text()):return False,'run_config_mismatch'
   if job.get('config_sha256') and digest(source)!=job['config_sha256']:return False,'plan_config_mismatch'
  value=json.loads((run/'validation_metrics.json').read_text())
  if value.get('split')!='val' or not value.get('reloaded_checkpoint_verified'):
   return False,'validation_reload_unverified'
  if not isinstance(value.get('epoch'),int) or not 1<=value['epoch']<=epochs:
   return False,'invalid_selected_epoch'
  if not math.isfinite(value['loss']) or value['loss']<0:return False,'invalid_validation_loss'
  for name in ('mIoU','mPA','mDice'):
   if not math.isfinite(value[name]) or not 0<=value[name]<=1:return False,'invalid_validation_metric:'+name
  metrics=metric_values(value,job['task'])
  if not all(math.isfinite(v) for v in metrics.values()):return False,'nonfinite_validation_metrics'
  return True,'validated'
 except (OSError,ValueError,TypeError,KeyError,AttributeError,AssertionError) as error:
  return False,'invalid_artifact:'+type(error).__name__

def predecessor_ready(root,plan):
 path=root/'reports/two_task/queue_status.json'
 if not path.exists():return False,'previous_queue_status_missing'
 try:state=json.loads(path.read_text())
 except (OSError,ValueError):return False,'previous_queue_status_invalid'
 if not isinstance(state,dict):return False,'previous_queue_status_invalid'
 if state.get('status')!='completed' or state.get('failures'):return False,'previous_queue_'+state.get('status','unknown')
 try:previous=json.loads((root/plan['previous_plan']).read_text())
 except (OSError,ValueError):return False,'previous_plan_invalid'
 if not isinstance(previous,dict) or not isinstance(previous.get('jobs'),list):return False,'previous_plan_invalid'
 if len(previous['jobs'])!=90:return False,'unexpected_previous_plan_size'
 for j in previous['jobs']:
  run=root/'two_task_runs'/f'twotask_{j["task"]}_{j["variant"]}'/f'seed_{j["seed"]}'
  valid,reason=validated_run(root,run,j,previous.get('epochs',150))
  if not valid:return False,'previous_'+reason
 return True,'all_90_validated'

def environment_record():
 names=['torch','torchvision','timm','mmcv','mmengine','natten','mamba-ssm','selective-scan','einops','numpy']
 sites={Path(sysconfig.get_path(k)) for k in ('purelib','platlib')}
 binaries={}
 for site in sorted(sites):
  files=list(site.glob('selective_scan*.so'))+list((site/'natten').rglob('*.so'))
  for path in files:binaries[str(path.relative_to(site))]=digest(path)
 return {'python':sys.version,'packages':{n:importlib.metadata.version(n) for n in names},'binaries':binaries}

def freeze(plan):
 old=json.loads((ROOT/'reports/two_task/runtime_lock.json').read_text());check_sources(old['sources'])
 assert digest(ROOT/plan['previous_plan'])==plan['previous_plan_sha256']==old['plan_sha256']
 configs={j['config']:j['config_sha256'] for j in all_jobs()}
 for name,h in configs.items():assert digest(ROOT/name)==h,'Config changed: '+name
 for j in plan['jobs']:
  proof=json.loads((FOLDER/f'setup_{j["task"]}_{j["model"]}.json').read_text())
  assert proof['passed'] and proof['config_sha256']==j['config_sha256'],'Setup check missing or stale'
  cfg=json.loads((ROOT/j['config']).read_text());m=cfg['model']
  if m.get('weights_path'):assert digest(ROOT/m['weights_path'])==m['weights_sha256']
 value={'plan_sha256':digest(ROOT/'configs/comparison_study/plan.json'),'sources':runtime_hashes(),
   'environment':environment_record(),'configs':configs,
   'data_manifests':{p:digest(ROOT/p) for p in ['data/processed/soldef_task1/manifest.json','data/processed/soldef/manifest.json']},
   'previous_dispatcher_sources':{str(p.relative_to(ROOT)):digest(p) for p in (ROOT/'two_task_study').glob('*.py')}}
 path=FOLDER/'runtime_lock.json'
 if path.exists():assert json.loads(path.read_text())==value,'Prepared sources/environment changed; refusing silent cohort changes'
 else:write_json(path,value)
 return value

def check_frozen(frozen):
 check_sources(frozen['sources'])
 check_sources(frozen['previous_dispatcher_sources'])
 for p,h in {**frozen['configs'],**frozen['data_manifests']}.items():assert digest(ROOT/p)==h,'Frozen input changed: '+p
 assert digest(ROOT/'configs/comparison_study/plan.json')==frozen['plan_sha256']
 assert environment_record()==frozen['environment'],'Environment changed'

def run_command(args,log,allow_busy=False):
 with log.open('a') as stream:result=subprocess.run([str(PYTHON),'-m',*args],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
 if allow_busy and result.returncode==75:return False
 if result.returncode:raise RuntimeError(f'Worker exited {result.returncode}; see {log}')
 return True

def execute(plan,frozen,state):
 check_frozen(frozen)
 for j in plan['jobs']:
  tag=j['task']+'_'+j['model'];proof=FOLDER/f'preflight_{tag}.json'
  if proof.exists():
   v=json.loads(proof.read_text());assert v['passed'] and len(v['classes'])==(3 if j['task']=='task1' else 5);continue
  state.update(phase='preflight',current=tag);write_json(FOLDER/'queue_status.json',state)
  run_command(['comparison_study.preflight','--config',j['config']],FOLDER/f'preflight_{tag}.log')
 for j in plan['jobs']:
  check_frozen(frozen);run=ROOT/j['run'];s=json.loads((run/'status.json').read_text()) if (run/'status.json').exists() else {}
  if s.get('evaluation_completed'):
   valid,reason=validated_run(ROOT,run,j,plan['epochs'])
   if not valid:raise RuntimeError('Invalid completed run '+j['run']+': '+reason)
   continue
  if shutil.disk_usage(ROOT).free<12*1024**3:raise RuntimeError('Less than 12 GiB free disk space')
  state.update(phase='training',current=j['run']);write_json(FOLDER/'queue_status.json',state)
  params=['--config',j['config'],'--seed',str(j['seed'])];log=FOLDER/f'{j["task"]}_{j["model"]}_{j["seed"]}.log'
  if s.get('status')!='completed':
   resume=['--resume'] if (run/'last.pt').exists() else []
   if run.exists() and not resume:raise RuntimeError('Existing unfinished run without resume checkpoint: '+j['run'])
   run_command(['comparison_study.train','train',*params,*resume],log)
  run_command(['comparison_study.train','evaluate',*params],log)
  valid,reason=validated_run(ROOT,run,j,plan['epochs'])
  if not valid:raise RuntimeError('Validation artifacts incomplete for '+j['run']+': '+reason)
  if (run/'last.pt').exists():(run/'last.pt').unlink()
  generate()
 check_frozen(frozen)
 from comparison_study.finalize import freeze_selection
 freeze_selection()
 state.update(phase='test',test_split_accessed=True);write_json(FOLDER/'queue_status.json',state)
 for j in all_jobs():
  check_frozen(frozen);state['current']=j['run'];write_json(FOLDER/'queue_status.json',state)
  run_command(['comparison_study.finalize','--run',j['run']],FOLDER/'test_evaluation.log');generate()
 state.update(phase='profiling');write_json(FOLDER/'queue_status.json',state)
 seen=set()
 for j in all_jobs():
  key=(j['task'],j['model'])
  if key in seen:continue
  seen.add(key);path=FOLDER/'profiles'/f'{key[0]}_{key[1]}.json'
  if path.exists():
   v=json.loads(path.read_text())
   if v.get('latency_valid'):continue
  state['current']='profile_'+key[0]+'_'+key[1];write_json(FOLDER/'queue_status.json',state)
  while not run_command(['comparison_study.profile','--task',key[0],'--model',key[1]],FOLDER/'profiling.log',allow_busy=True):
   state['note']='Waiting for idle GPU to measure comparable FPS';write_json(FOLDER/'queue_status.json',state);time.sleep(60)
  generate()
 state.update(status='completed',phase='finished',current=None);state.pop('note',None);write_json(FOLDER/'queue_status.json',state);generate()

def main():
 p=argparse.ArgumentParser();p.add_argument('--prepare-only',action='store_true');p.add_argument('--report-only',action='store_true');a=p.parse_args()
 plan=json.loads((ROOT/'configs/comparison_study/plan.json').read_text());assert len(plan['jobs'])==60
 FOLDER.mkdir(parents=True,exist_ok=True)
 if a.report_only:generate();return
 with (FOLDER/'supervisor.lock').open('a') as supervisor:
  fcntl.flock(supervisor,fcntl.LOCK_EX|fcntl.LOCK_NB);frozen=freeze(plan)
  if a.prepare_only:generate();print('Frozen 60-job successor');return
  state={'pid':os.getpid(),'status':'waiting_for_predecessor','phase':'waiting','planned':60,'tasks':['task1','task2'],
    'test_split_accessed':False,'previous_plan_jobs':90,'current':None,'failures':[]}
  write_json(FOLDER/'queue_status.json',state);generate()
  try:
   while True:
    ready,reason=predecessor_ready(ROOT,plan);state['waiting_reason']=reason;write_json(FOLDER/'queue_status.json',state)
    if ready:
     with contextlib.ExitStack() as stack:
      try:
       for path in [ROOT/'reports/two_task/queue.lock',ROOT/'reports/paper_protocol/queue.lock']:
        path.parent.mkdir(parents=True,exist_ok=True)
        handle=stack.enter_context(path.open('a'));fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
      except BlockingIOError:pass
      else:
       ready,reason=predecessor_ready(ROOT,plan)
       if ready:
        state.update(status='running');state.pop('waiting_reason',None);execute(plan,frozen,state);return
    generate();time.sleep(30)
  except BaseException as error:
   state.update(status='failed',error=repr(error));state['failures'].append({'current':state.get('current'),'error':repr(error)})
   write_json(FOLDER/'queue_status.json',state);generate();raise
if __name__=='__main__':main()
