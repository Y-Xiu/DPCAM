"""Plan 60 author-model comparisons following the existing 90 ablations."""
import json
from pathlib import Path
from common import ROOT,digest,write_json
NAMES=('emcad','segman_s','mambavision_t','segformer_b2','unet2015','deeplabv3_r101_paper')
WEIGHTS={'emcad':'pvt_v2_b2.pth','segman_s':'SegMAN_Encoder_s.pth.tar','mambavision_t':'mambavision_tiny_1k.pth.tar','segformer_b2':'mit_b2.pth'}
# Planning is offline; the dispatcher and model loader verify these exact files.
WEIGHT_SHA256={
 'emcad':'80711cd1b37ffba12bec6c7a2a7c54efe2315ed635e6d2055fd51c3e909ede4d',
 'segman_s':'be12d9954e49a12266913018d5565261825c53c530cf8a0ba0fdb05e4ea33059',
 'mambavision_t':'952a3e486f94bbe863c753a7ecabe282b2e3b8adbb0d98057e047e4f554c2a9b',
 'segformer_b2':'7cd40972ccdb467b3e061ec99e29470dee0245ac684031f01ca64af0427e44c0',
}
DEEPLAB_MODEL={'name':'deeplabv3_r101_paper','backbone':'resnet101','attention':None,
 'pretrained':True,'weights_path':'weights/resnet101-63fe2227.pth',
 'weights_sha256':'63fe2227b86e8f1f2063f43a75c84d195911b6a0eace650907dd3dc62dd49a0a'}
FOLDER=ROOT/'configs/comparison_study'
def main():
 FOLDER.mkdir(parents=True,exist_ok=True);jobs=[]
 for seed in range(42,47):
  for name in NAMES:
   for task in ('task1','task2'):
    cfg=json.loads((ROOT/f'configs/two_task/twotask_{task}_none.json').read_text())
    cfg.update(protocol_id='two_task_comparison_v1',experiment=f'comparison_{task}_{name}',
      evaluation_scope='train/validation; all comparisons plus ablations selected before final test',
      initialization_protocol='ImageNet-1k encoder, random segmentation heads; classic U-Net from scratch',
      model_provenance='Pinned author model; modern_study/vendor. Necessary author auxiliary supervision retained; common SolDef training budget, not original benchmark replication.')
    cfg.pop('study_variant');cfg.pop('position_encoding',None)
    cfg['model']={'name':name,'pretrained':name!='unet2015'}
    if name in WEIGHTS:
     cfg['model'].update(weights_path='modern_study/assets/'+WEIGHTS[name],weights_sha256=WEIGHT_SHA256[name])
    if name=='deeplabv3_r101_paper':cfg['model']=dict(DEEPLAB_MODEL)
    cfg['training_loss']='author_mutation_15_subsets_sum_0.3CE_0.7Dice' if name=='emcad' else ('author_CE_main_plus_0.4CE_aux' if name=='mambavision_t' else ('author_CE' if name in ('segman_s','segformer_b2') else 'stable_paper_DiceFocal'))
    cfg['validation_selection_loss']='common stable paper Dice/Focal on main output only'
    path=FOLDER/(cfg['experiment']+'.json')
    if path.exists():assert json.loads(path.read_text())==cfg,'Refuse to change prepared config'
    else:write_json(path,cfg)
    jobs.append({'task':task,'model':name,'seed':seed,'config':str(path.relative_to(ROOT)),
       'config_sha256':digest(path),'run':f'comparison_runs/{cfg["experiment"]}/seed_{seed}'})
 value={'protocol_id':'two_task_comparison_v1','previous_plan':'configs/two_task/plan.json','previous_plan_sha256':digest(ROOT/'configs/two_task/plan.json'),
 'models':list(NAMES),'tasks':['task1','task2'],'seeds':list(range(42,47)),'epochs':150,'jobs':jobs,
 'order':'seed, model, alternating task1/task2','start_gate':'previous queue lock released, status completed with no failures, all 90 evaluation_completed',
 'metrics':['mIoU_percent','mPA_percent','Recall_defect_percent','F1_macro','Parameters_M','FLOPs_supported_G','FPS'],
 'test_gate':'all 90 ablations plus 60 comparisons validated; freeze all selected checkpoint hashes before test'}
 path=FOLDER/'plan.json'
 if path.exists():assert json.loads(path.read_text())==value
 else:write_json(path,value)
 print('Prepared',len(jobs),'runs')
if __name__=='__main__':main()
