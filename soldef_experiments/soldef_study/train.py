"""SolDef-only 150-epoch development: matched initialization, ablations and separate candidates.
Derived from the locked paper runner; separate protocol, outputs and source snapshots.
"""
import argparse
import fcntl
import json
import shutil
import time
import math
from pathlib import Path
import numpy as np
import torch
from common import ROOT, digest, set_seed, write_json
from experiment import architecture_info, device_for, environment, epoch_pass
from innovation_study.stable_loss import StablePaperLoss as PaperLoss
from soldef_study.model import create_model,assert_architecture
from paper_protocol.data import audit, make_loader


def runtime_hashes():
    files=[ROOT/n for n in ['common.py','dataset.py','experiment.py','losses.py','metrics.py','prepare_data.py']]
    files += [ROOT/'innovation_study'/n for n in ['stable_loss.py','residual_candidate.py']]
    files += list((ROOT/'soldef_study').glob('*.py'))
    files += list((ROOT/'two_task_study').glob('*.py'))
    files+=list((ROOT/'models').rglob('*.py'))+list((ROOT/'paper_protocol').glob('*.py'))
    files+=list((ROOT/'third_party').rglob('*.py'))
    return {str(p.relative_to(ROOT)):digest(p) for p in sorted(set(files)) if '.git' not in p.parts}


def check_sources(hashes):
    for name,h in hashes.items():
        if digest(ROOT/name)!=h:raise ValueError('Runtime changed: '+name)


def save(value,path):
    tmp=path.with_suffix('.tmp');torch.save(value,tmp);tmp.replace(path)


def run_root(config):
    return ROOT/('two_task_runs' if config['protocol_id']=='two_task_v1' else 'soldef_runs')


def validate_config(config):
    if config['protocol_id'] not in ('soldef_focus_v1','two_task_v1'):raise ValueError('Wrong protocol')
    assert config['dataset'] in ('soldef','soldef_task1')
    assert config['study_variant'] in ('none','cbam','dpcam','no_dca','no_pa','no_dsa','equal_spatial','residual_cbam','residual_dpcam')
    t=config['training']
    if (config['image_size'],t['epochs'],t['batch_size'],t['lr'])!=(512,150,2,1e-4):
        raise ValueError('Unexpected SolDef focus input/training settings')
    if t['scheduler']!='cosine' or t['min_lr']!=1e-6 or t['mixed_precision'] or t['selection']!='min_val_loss':
        raise ValueError('Unexpected SolDef focus optimization/selection')
    assert t['optimizer']=='AdamW'
    assert config['model']['name']=='paper_unet' and config['model']['backbone']=='convnext_base'
    assert config['classes'][0]=='background' and len(set(config['classes']))==len(config['classes'])


def load_config(path):
    config=json.loads(Path(path).read_text())
    validate_config(config)
    return config


def learning_rate(training,epoch):
    if training['epochs']<2:return training['lr']
    return training['min_lr']+.5*(training['lr']-training['min_lr'])*(1+math.cos(math.pi*epoch/(training['epochs']-1)))


def train(config,seed,device,resume=False,stop_after=None):
    validate_config(config);t=config['training'];run=run_root(config)/config['experiment']/f'seed_{seed}'
    data=audit(ROOT/config['manifest'])
    if data['manifest_sha256']!=config['manifest_sha256']:raise ValueError('Frozen data changed')
    stored=None
    if resume:
        stored=torch.load(run/'last.pt',map_location='cpu',weights_only=True)
        if stored['config']!=config or stored['seed']!=seed:raise ValueError('Resume configuration mismatch')
        check_sources(stored['source_sha256'])
        assert stored['manifest_sha256']==data['manifest_sha256']
    else:
        run.mkdir(parents=True,exist_ok=False)
        write_json(run/'config.json',config)
        shutil.copy2(ROOT/config['manifest'],run/'data_manifest.json')
    status={'status':'running','protocol_id':config['protocol_id'],'seed':seed,'test_split_accessed':False}
    write_json(run/'status.json',status)
    try:
        set_seed(seed);torch.set_num_threads(t['cpu_threads'])
        model=create_model(config,pretrained=False if resume else None,seed=seed).to(device)
        assert_architecture(config,model)
        optimizer=torch.optim.AdamW(model.parameters(),lr=t['lr'],weight_decay=t['weight_decay'])
        if stored:
            model.load_state_dict(stored['model_state_dict'],strict=True)
            optimizer.load_state_dict(stored['optimizer_state_dict'])
            history,best,hashes=stored['history'],stored['best'],stored['source_sha256']
            start_epoch,elapsed_before=stored['epoch'],stored['elapsed_seconds']
            del stored
        else:
            hashes=runtime_hashes();history=[];best={'loss':float('inf'),'epoch':0};start_epoch=0;elapsed_before=0.
            for name in hashes:
                dst=run/'source'/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dst)
            arch=architecture_info(model)
            encoder=getattr(model,'encoder',None)
            pretrained_cfg=getattr(encoder,'pretrained_cfg',None)
            arch['encoder_pretrained_cfg']=pretrained_cfg
            write_json(run/'architecture.json',arch)
            write_json(run/'provenance.json',{'source_sha256':hashes,'environment':environment(),
                       'manifest_sha256':data['manifest_sha256'],'data_audit':data,'seed':seed,
                       'initialization':arch['initialization'],'encoder_pretrained_cfg':pretrained_cfg})
        status.update(epoch=start_epoch,best=best if start_epoch else None,elapsed_seconds=elapsed_before)
        criterion=PaperLoss(**config['loss']);started=time.perf_counter()
        limit=min(stop_after or t['epochs'],t['epochs'])
        for epoch in range(start_epoch,limit):
            set_seed(seed+epoch)
            lr=learning_rate(t,epoch)
            for group in optimizer.param_groups:group['lr']=lr
            training,_=epoch_pass(model,make_loader(config,'train',seed+epoch),criterion,device,optimizer,classes=config['classes'])
            validation,_=epoch_pass(model,make_loader(config,'val',seed),criterion,device,classes=config['classes'])
            elapsed=elapsed_before+time.perf_counter()-started
            history.append({'epoch':epoch+1,'lr':lr,'train':training,'val':validation,'elapsed_seconds':elapsed})
            checkpoint={'config':config,'seed':seed,'epoch':epoch+1,'manifest_sha256':data['manifest_sha256'],
                        'source_sha256':hashes,'model_state_dict':model.state_dict()}
            if validation['loss']<best['loss']:
                best={'loss':validation['loss'],'epoch':epoch+1}
                save({**checkpoint,'val':validation},run/'best.pt')
            save({**checkpoint,'optimizer_state_dict':optimizer.state_dict(),'history':history,'best':best,
                  'elapsed_seconds':elapsed},run/'last.pt')
            write_json(run/'history.json',history)
            status.update(epoch=epoch+1,best=best,elapsed_seconds=elapsed)
            write_json(run/'status.json',status)
            print(f"{config['experiment']} seed={seed} epoch={epoch+1}/{t['epochs']} train={training['loss']:.6f} val={validation['loss']:.6f} mIoU={validation['mIoU']:.4f} fg={validation['foreground_mIoU']:.4f} elapsed={elapsed:.1f}s",flush=True)
        status.update(status='completed' if limit==t['epochs'] else 'paused',stopping_reason='fixed_150_epochs' if limit==t['epochs'] else 'explicit_pause')
        write_json(run/'status.json',status)
    except BaseException as exc:
        status.update(status='failed',error=repr(exc));write_json(run/'status.json',status);raise
    return run


def evaluate(run,device):
    run=Path(run);status=json.loads((run/'status.json').read_text())
    if status['status']!='completed' or status['epoch']!=150:raise ValueError('Full 150 epochs required before evaluation')
    ckpt=torch.load(run/'best.pt',map_location='cpu',weights_only=True);cfg=ckpt['config'];validate_config(cfg)
    check_sources(ckpt['source_sha256']);data=audit(ROOT/cfg['manifest'])
    if data['manifest_sha256']!=ckpt['manifest_sha256']:raise ValueError('Data changed')
    torch.set_num_threads(cfg['training']['cpu_threads']);set_seed(ckpt['seed'])
    model=create_model(cfg,pretrained=False,seed=ckpt['seed']).to(device);model.load_state_dict(ckpt['model_state_dict'],strict=True)
    assert_architecture(cfg,model);criterion=PaperLoss(**cfg['loss'])
    validation,matrix=epoch_pass(model,make_loader(cfg,'val',ckpt['seed']),criterion,device,classes=cfg['classes'])
    for k in ['loss','mIoU','mDice','mPA']:
        if not np.isclose(validation[k],ckpt['val'][k],atol=1e-7,rtol=1e-6):raise ValueError('Reloaded validation differs: '+k)
    write_json(run/'validation_metrics.json',{'epoch':ckpt['epoch'],'split':'val','reloaded_checkpoint_verified':True,**validation})
    np.savetxt(run/'validation_confusion.csv',matrix,delimiter=',',fmt='%d')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    history=json.loads((run/'history.json').read_text())
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    axes[0].plot([h['epoch'] for h in history],[h['train']['loss'] for h in history],label='train')
    axes[0].plot([h['epoch'] for h in history],[h['val']['loss'] for h in history],label='validation')
    axes[0].axvline(ckpt['epoch'],color='black',linestyle='--',label='selected checkpoint')
    axes[0].set(xlabel='Epoch',ylabel='Dice/Focal loss');axes[0].legend()
    norm=np.divide(matrix,matrix.sum(1,keepdims=True),out=np.zeros_like(matrix,dtype=float),where=matrix.sum(1,keepdims=True)>0)
    axes[1].imshow(norm,vmin=0,vmax=1,cmap='Blues')
    axes[1].set(xticks=range(len(cfg['classes'])),yticks=range(len(cfg['classes'])),xticklabels=cfg['classes'],yticklabels=cfg['classes'],xlabel='Predicted',ylabel='Ground truth',title='Validation confusion (row normalized)')
    axes[1].tick_params(axis='x',rotation=60)
    for i in range(len(cfg['classes'])):
        for j in range(len(cfg['classes'])):axes[1].text(j,i,f'{norm[i,j]:.2f}',ha='center',va='center',color='white' if norm[i,j]>.5 else 'black',fontsize=8)
    fig.tight_layout();fig.savefig(run/'curves_and_confusion.png',dpi=160);plt.close(fig)
    status.update(test_split_accessed=False,validation_reload_verified=True,evaluation_completed=True,development_only=True)
    write_json(run/'status.json',status)
    print('EVALUATED '+str(run)+' '+json.dumps({'val_mIoU':validation['mIoU'],'test_split_accessed':False}),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['train','evaluate']);p.add_argument('--config',required=True)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--resume',action='store_true');p.add_argument('--stop-after',type=int)
    p.add_argument('--device',default='cuda',choices=['cuda','cpu']);args=p.parse_args()
    cfg=load_config(args.config);run=run_root(cfg)/cfg['experiment']/f'seed_{args.seed}'
    lockpath=ROOT/('reports/two_task' if cfg['protocol_id']=='two_task_v1' else 'reports/soldef_focus')/f'{cfg["experiment"]}_{args.seed}.lock';lockpath.parent.mkdir(parents=True,exist_ok=True)
    with lockpath.open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.action=='train':train(cfg,args.seed,device_for(args.device),args.resume,args.stop_after)
        else:evaluate(run,device_for(args.device))

if __name__=='__main__':main()
