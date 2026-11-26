"""Task-aware metrics from completed runs; no GPU or test-set access."""
import json,statistics,math
from pathlib import Path
from common import ROOT,write_json
FOLDER=ROOT/'reports/comparison_study'
LABELS={'emcad':'EMCAD (PVTv2-B2)','segman_s':'SegMAN-S','mambavision_t':'MambaVision-T + UPerNet','segformer_b2':'SegFormer-B2','unet2015':'U-Net','deeplabv3_r101_paper':'DeepLabv3 (ResNet-101)',
 'none':'ConvNeXt-Base + U-Net','cbam':'ConvNeXt-Base + CBAM','dpcam':'ConvNeXt-Base + DPCAM','no_dca':'DPCAM w/o DCA','no_pa':'DPCAM w/o PA','no_dsa':'DPCAM w/o DSA','equal_spatial':'DPCAM equal spatial weights','residual_cbam':'Residual CBAM (candidate)','residual_dpcam':'Residual DPCAM (candidate)'}
def metric_values(v,task):
 if task not in ('task1','task2'):raise ValueError('Unknown task')
 per=v['per_class'];expected=['background','good','no_good'] if task=='task1' else ['background','good','exc_solder','poor_solder','spike']
 assert set(per)==set(expected)
 for value in [v[k] for k in ('mIoU','mPA','mDice')]+[m[k] for m in per.values() for k in ('IoU','Recall','Dice')]:
  if not isinstance(value,(int,float)) or not math.isfinite(value) or not 0<=value<=1:raise ValueError('Nonfinite or out-of-range metric')
 for macro,field in [('mIoU','IoU'),('mPA','Recall'),('mDice','Dice')]:
  assert abs(v[macro]-sum(p[field] for p in per.values())/len(per))<1e-8
 defects=['no_good'] if task=='task1' else ['exc_solder','poor_solder','spike']
 return {'mIoU_percent':100*v['mIoU'],'mPA_percent':100*v['mPA'],
 'Recall_defect_percent':100*sum(per[c]['Recall'] for c in defects)/len(defects),'F1_macro':v['mDice']}
def read_artifact(path, errors):
 try:
  value=json.loads(path.read_text())
  if not isinstance(value,dict):raise ValueError('Expected JSON object')
  return value
 except (OSError,ValueError) as error:
  errors.append({'file':str(path),'error':str(error)})
  return None


def all_jobs():
 plan=json.loads((ROOT/'configs/comparison_study/plan.json').read_text());jobs=[{**j,'kind':'comparison'} for j in plan['jobs']]
 old=json.loads((ROOT/plan['previous_plan']).read_text())
 for j in old['jobs']:jobs.append({**j,'model':j['variant'],'kind':'ablation','run':f'two_task_runs/twotask_{j["task"]}_{j["variant"]}/seed_{j["seed"]}'})
 return jobs

def generate():
 jobs=all_jobs();rows=[];errors=[]
 FOLDER.mkdir(parents=True,exist_ok=True)
 for j in jobs:
  run=ROOT/j['run'];status=(read_artifact(run/'status.json',errors) or {'status':'invalid'}) if (run/'status.json').exists() else {'status':'pending'}
  r={**j,'status':status}
  if status.get('status')=='completed' and status.get('epoch')==150 and status.get('evaluation_completed') and status.get('validation_reload_verified'):
   for split in ['validation','test']:
    path=run/f'{split}_metrics.json'
    if path.exists():
     value=read_artifact(path,errors)
     if value is not None:
      try:
       metric_values(value,j['task'])
       assert value.get('split')==('val' if split=='validation' else 'test')
       if split=='validation':assert value.get('reloaded_checkpoint_verified')
       r[split]=value
      except (AssertionError,KeyError,TypeError,ValueError) as error:errors.append({'file':str(path),'error':'Invalid metric record: '+repr(error)})
  p=run/'architecture.json'
  if p.exists():
   value=read_artifact(p,errors)
   if value is not None and isinstance(value.get('parameters'),int):r['parameters']=value['parameters']
  rows.append(r)
 write_json(FOLDER/'results.json',{'runs':rows,'invalid_artifacts':errors,'comparison_validated':sum(r['kind']=='comparison' and 'validation' in r for r in rows)})
 tables={};lines=['# 两任务新增对比与消融结果','',
 '现有90项消融结束后接续60项新对比。所有模型采用同一任务的原图分组划分、512输入、150轮、五种子；保留作者必要训练损失/辅助监督，统一以验证集主输出的稳定Dice/Focal最低值选模。验证集与测试集分别列出；未完成项留空。',
 'mIoU、mPA包含背景；Task 1的Recall_defect=no_good召回率，Task 2为exc_solder/poor_solder/spike三类宏召回率；F1_macro为全部类别逐类F1的平均，0–1。±为独立种子间样本标准差。','',
 '参数包括辅助头；FLOPs_supported仅计PyTorch profiler支持算子，乘加计2，缺失SSM/NATTEN/融合注意力等，不能当作完整模型FLOPs。FPS统一batch1/512/FP32前向，30次预热、200次计时，空闲GPU测量。',
 'Task 1有228原图、456重叠裁剪，318/68/70；Task 2有200原图、400裁剪，280/60/60；二者原图不跨集合。Task 1裁剪截断问题见此前独立审计，本轮保留现有划分/裁剪保证与消融同条件。','']
 main=['unet2015','deeplabv3_r101_paper','segformer_b2','emcad','segman_s','mambavision_t','none','dpcam']
 abl=['none','cbam','dpcam','no_dca','no_pa','no_dsa','equal_spatial','residual_cbam','residual_dpcam']
 for task in ('task1','task2'):
  for split in ('validation','test'):
   for kind,models in [('comparison',main),('ablation',abl)]:
    tag=f'{task}_{kind}_{split}';table=[];lines += [f'## {tag}','','| Model | n | mIoU (%) | mPA (%) | Recall_defect (%) | F1_macro | Params (M) | FLOPs_supported (G) | FPS |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    tex=[r'\begin{tabular}{lrrrrrrrr}',r'\toprule',r'Model & n & mIoU (\%) & mPA (\%) & Recall$_{defect}$ (\%) & F1$_{macro}$ & Params (M) & FLOPs$_{supported}$ (G) & FPS \\',r'\midrule']
    for model in models:
     subset=[r for r in rows if r['task']==task and r['model']==model and split in r]
     assert len({r['seed'] for r in subset})==len(subset)
     values=[metric_values(r[split],task) for r in subset];summary={}
     if values:
      for k in values[0]:
       vals=[v[k] for v in values];summary[k]={'mean':statistics.mean(vals),'sample_std':statistics.stdev(vals) if len(vals)>1 else None}
     def fmt(k):
      if k not in summary:return '—'
      s=summary[k];return f"{s['mean']:.3f} ± {s['sample_std']:.3f}" if s['sample_std'] is not None else f"{s['mean']:.3f} (n=1)"
     counts={r['parameters'] for r in rows if r['task']==task and r['model']==model and 'parameters' in r};assert len(counts)<=1
     path=FOLDER/'profiles'/f'{task}_{model}.json';profile=(read_artifact(path,errors) or {}) if path.exists() else {}
     params=next(iter(counts)) if counts else profile.get('parameters')
     efficiency=[f'{params/1e6:.3f}' if params else '—',f"{profile['supported_operator_GFLOPs']:.3f}" if 'supported_operator_GFLOPs' in profile else '—',f"{profile['fps']:.3f}" if profile.get('latency_valid') else '—']
     cells=[LABELS[model],str(len(subset)),*[fmt(k) for k in ('mIoU_percent','mPA_percent','Recall_defect_percent','F1_macro')],*efficiency]
     lines.append('| '+' | '.join(cells)+' |');tex.append(' & '.join(x.replace('±',r'$\pm$').replace('—','--') for x in cells)+r' \\')
     table.append({'model':model,'n':len(subset),'accuracy':summary,'parameters':params,'profile':profile})
    tex += [r'\bottomrule',r'\end{tabular}'];(FOLDER/(tag+'.tex')).write_text('\n'.join(tex)+'\n');tables[tag]=table
 if errors:lines += ['','## Invalid / incomplete artifacts','']+['- '+e['file']+': '+e['error'] for e in errors]
 write_json(FOLDER/'artifact_errors.json',errors)
 write_json(FOLDER/'tables.json',tables);(FOLDER/'RESULTS.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':generate()
