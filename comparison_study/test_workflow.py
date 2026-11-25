"""Meaningful gates, task-aware metrics, and deterministic epoch-boundary resume."""
import json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from comparison_study.suite import predecessor_ready,validated_run
from comparison_study import suite,prepare
from common import ROOT,digest
from comparison_study.report import metric_values
from comparison_study import train as runner
from metrics import ConfusionMatrix

class WorkflowTests(unittest.TestCase):
 def test_predecessor_requires_all_validations(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);(root/'reports/two_task').mkdir(parents=True);(root/'configs').mkdir()
   jobs=[{'task':'task1','variant':'v'+str(i),'seed':42} for i in range(90)]
   (root/'configs/old.json').write_text(json.dumps({'jobs':jobs}));plan={'previous_plan':'configs/old.json'}
   state=root/'reports/two_task/queue_status.json';state.write_text(json.dumps({'status':'running','failures':[]}))
   self.assertFalse(predecessor_ready(root,plan)[0]);state.write_text(json.dumps({'status':'completed','failures':[]}))
   self.assertFalse(predecessor_ready(root,plan)[0])
   for j in jobs:
    run=root/'two_task_runs'/f'twotask_task1_{j["variant"]}'/'seed_42';run.mkdir(parents=True)
    (run/'status.json').write_text(json.dumps({'status':'completed','epoch':150,'evaluation_completed':True,'validation_reload_verified':True}))
    (run/'best.pt').write_bytes(b'checkpoint fixture')
    (run/'config.json').write_text('{}')
    matrix=ConfusionMatrix(3);matrix.matrix=np.diag([10,8,6])
    value={'epoch':120,'split':'val','reloaded_checkpoint_verified':True,'loss':.2,
     **matrix.compute(['background','good','no_good'])}
    (run/'validation_metrics.json').write_text(json.dumps(value))
   self.assertTrue(predecessor_ready(root,plan)[0])
   # A completed status alone must never release the next training cohort.
   metrics=run/'validation_metrics.json';original=metrics.read_text()
   for broken in ('', '{', json.dumps({**value,'mIoU':float('nan')}),
                  json.dumps({**value,'split':'test'}),json.dumps({**value,'mPA':.5})):
    metrics.write_text(broken);self.assertFalse(predecessor_ready(root,plan)[0])
   metrics.write_text(original);self.assertTrue(predecessor_ready(root,plan)[0])
   (run/'best.pt').write_bytes(b'');self.assertFalse(predecessor_ready(root,plan)[0])
   (run/'best.pt').write_bytes(b'checkpoint fixture')
   (run/'status.json').write_text('');self.assertFalse(predecessor_ready(root,plan)[0])
   state.write_text(json.dumps({'status':'completed_with_failures','failures':['error']}))
   self.assertFalse(predecessor_ready(root,plan)[0])
 def test_offline_plan_preserves_published_configs(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);(root/'configs/two_task').mkdir(parents=True)
   for name in ('plan.json','twotask_task1_none.json','twotask_task2_none.json'):
    (root/'configs/two_task'/name).write_bytes((ROOT/'configs/two_task'/name).read_bytes())
   with patch.object(prepare,'ROOT',root),patch.object(prepare,'FOLDER',root/'configs/comparison_study'):
    prepare.main()
   # No vendor, weights, old crop-level configs, or local environment was copied.
   for path in (ROOT/'configs/comparison_study').glob('*.json'):
    self.assertEqual(json.loads(path.read_text()),json.loads((root/'configs/comparison_study'/path.name).read_text()),path.name)
 def test_dispatcher_uses_current_interpreter_and_native_extensions(self):
  self.assertEqual(suite.PYTHON,Path(sys.executable))
  with tempfile.TemporaryDirectory() as td:
   site=Path(td);(site/'natten/lib').mkdir(parents=True)
   extension=site/'natten/lib/_C.so';extension.write_bytes(b'native library fixture')
   with patch.object(suite.sysconfig,'get_path',return_value=str(site)),patch.object(suite.importlib.metadata,'version',return_value='fixture'):
    value=suite.environment_record()
   self.assertEqual(value['binaries'],{'natten/lib/_C.so':digest(extension)})
 def test_three_and_five_class_metrics_are_not_double_scaled(self):
  for task,names,matrix in [('task1',['background','good','no_good'],np.array([[8,1,1],[1,7,2],[2,1,7]])),
    ('task2',['background','good','exc_solder','poor_solder','spike'],np.diag([9,8,7,6,5]))]:
   cm=ConfusionMatrix(len(names));cm.matrix=matrix;v=cm.compute(names);result=metric_values(v,task)
   self.assertAlmostEqual(result['mPA_percent'],100*np.mean(np.diag(matrix)/matrix.sum(1)))
   self.assertTrue(0<=result['F1_macro']<=1);self.assertTrue(0<=result['mIoU_percent']<=100)
   if task=='task1':self.assertAlmostEqual(result['Recall_defect_percent'],70)
 def test_resuming_preserves_parameters_optimizer_and_validation(self):
  torch.set_num_threads(1)
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);(root/'manifest.json').write_text('{}')
   base={'experiment':'full','protocol_id':'two_task_comparison_v1','dataset':'soldef_task1','task':'task1',
    'classes':['background','good','no_good'],'image_size':512,'split_protocol':'source_group','manifest':'manifest.json','manifest_sha256':'fixture',
    'model':{'name':'unet2015','pretrained':False},'training_loss':'stable',
    'training':{'epochs':150,'cpu_threads':1,'batch_size':2,'lr':1e-4,'min_lr':1e-6,'weight_decay':1e-5,'scheduler':'cosine','selection':'min_val_loss','mixed_precision':False},
    'loss':{'alpha':.25,'gamma':2.,'dice_weight':.5,'focal_weight':.5,'smooth':1e-6}}
   def make_model(*args):return torch.nn.Sequential(torch.nn.Conv2d(3,4,1),torch.nn.Dropout2d(.2),torch.nn.Conv2d(4,3,1))
   generator=torch.Generator().manual_seed(7);x=torch.randn(2,3,8,8,generator=generator);y=torch.randint(0,3,(2,8,8),generator=generator)
   def loader(*args):return [(x,y,['a','b'])]
   with patch.object(runner,'ROOT',root),patch.object(runner,'audit',return_value={'manifest_sha256':'fixture'}),patch.object(runner,'runtime_hashes',return_value={}),patch.object(runner,'environment',return_value={}),patch.object(runner,'create_model',side_effect=make_model),patch.object(runner,'make_loader',side_effect=loader):
    runner.train(base,42,torch.device('cpu'),stop_after=2)
    resumed={**base,'experiment':'resumed'};runner.train(resumed,42,torch.device('cpu'),stop_after=1);runner.train(resumed,42,torch.device('cpu'),resume=True,stop_after=2)
   a=torch.load(root/'comparison_runs/full/seed_42/last.pt',weights_only=True);b=torch.load(root/'comparison_runs/resumed/seed_42/last.pt',weights_only=True)
   for k in a['model_state_dict']:self.assertTrue(torch.equal(a['model_state_dict'][k],b['model_state_dict'][k]),k)
   self.assertEqual(a['best'],b['best'])
   for x,y in zip(a['history'],b['history']):self.assertEqual(x['val'],y['val']);self.assertEqual(x['train'],y['train'])
   for key,value in a['optimizer_state_dict']['state'].items():
    for n,t in value.items():self.assertTrue(torch.equal(t,b['optimizer_state_dict']['state'][key][n]),n)

if __name__=='__main__':unittest.main()
