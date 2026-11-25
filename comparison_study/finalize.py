"""Test only after all 150 selections are frozen; no test-driven decisions."""
import argparse,json
import numpy as np
import torch
from common import ROOT,digest,set_seed,write_json
from experiment import epoch_pass
from innovation_study.stable_loss import StablePaperLoss
from paper_protocol.data import audit,make_loader
from soldef_study.train import check_sources
from comparison_study.report import all_jobs
FOLDER=ROOT/'reports/comparison_study'
def freeze_selection():
 from comparison_study.suite import validated_run
 records=[]
 jobs=all_jobs()
 assert len(jobs)==150 and len({j['run'] for j in jobs})==150

 for j in jobs:
  run=ROOT/j['run']
  valid,reason=validated_run(ROOT,run,j)
  if not valid:raise ValueError('Cannot freeze incomplete selection: '+j['run']+': '+reason)
  records.append({'run':j['run'],'best_sha256':digest(run/'best.pt'),'validation_sha256':digest(run/'validation_metrics.json'),
    'config_sha256':digest(ROOT/j['config'])})
 value={'plan_sha256':digest(ROOT/'configs/comparison_study/plan.json'),'selection':'min validation loss; 150 epochs; all 150 runs frozen before test','runs':records}
 path=FOLDER/'selection_lock.json'
 if path.exists():assert json.loads(path.read_text())==value,'Selected cohort changed'
 else:write_json(path,value)
 return value

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args()
 jobs=all_jobs();job=next(j for j in jobs if j['run']==a.run);lock=json.loads((FOLDER/'selection_lock.json').read_text())
 assert len(lock['runs'])==150 and lock['plan_sha256']==digest(ROOT/'configs/comparison_study/plan.json')
 record=next(r for r in lock['runs'] if r['run']==a.run);run=ROOT/a.run
 for f,k in [('best.pt','best_sha256'),('validation_metrics.json','validation_sha256')]:assert digest(run/f)==record[k]
 assert digest(ROOT/job['config'])==record['config_sha256']
 target=run/'test_metrics.json'
 if target.exists():
  v=json.loads(target.read_text());assert v['checkpoint_sha256']==record['best_sha256'] and v['selection_lock_sha256']==digest(FOLDER/'selection_lock.json');return
 ckpt=torch.load(run/'best.pt',map_location='cpu',weights_only=True);cfg=ckpt['config'];check_sources(ckpt['source_sha256'])
 assert cfg==json.loads((ROOT/job['config']).read_text());assert audit(ROOT/cfg['manifest'])['manifest_sha256']==cfg['manifest_sha256']
 set_seed(job['seed']);torch.set_num_threads(cfg['training']['cpu_threads'])
 if job['kind']=='ablation':
  from soldef_study.model import create_model
  model=create_model(cfg,False,seed=job['seed'])
 else:
  from modern_study.model import create_model
  model=create_model(cfg,False)
 model.load_state_dict(ckpt['model_state_dict'],strict=True);model=model.cuda()
 # Verify validation reload under this interpreter before accessing test data.
 validation,_=epoch_pass(model,make_loader(cfg,'val',job['seed']),StablePaperLoss(**cfg['loss']),torch.device('cuda'),classes=cfg['classes'])
 previous=json.loads((run/'validation_metrics.json').read_text())
 for k in ['loss','mIoU','mPA','mDice']:assert np.isclose(validation[k],previous[k],rtol=1e-6,atol=1e-7),k
 result,cm=epoch_pass(model,make_loader(cfg,'test',job['seed']),StablePaperLoss(**cfg['loss']),torch.device('cuda'),classes=cfg['classes'],
     output_dir=run/'test_predictions' if job['seed']==42 else None)
 write_json(target,{'split':'test','epoch':ckpt['epoch'],'checkpoint_sha256':record['best_sha256'],
  'selection_lock_sha256':digest(FOLDER/'selection_lock.json'),'manifest_sha256':cfg['manifest_sha256'],**result})
 np.savetxt(run/'test_confusion.csv',cm,delimiter=',',fmt='%d')
 write_json(run/'final_evaluation_status.json',{'completed':True,'test_split_accessed':True,'checkpoint_sha256':record['best_sha256']})
 print('TEST',job['run'],result['mIoU'],flush=True)
if __name__=='__main__':main()
