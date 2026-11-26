"""Shared source-group data audit and loader; legacy module name kept for imports."""
import json
from collections import Counter, defaultdict
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from common import ROOT, SPLITS, digest, seed_worker, write_json
from dataset import SegmentationDataset


def audit(path):
    path=Path(path);m=json.loads(path.read_text())
    if m.get('split_unit') != 'source_image':raise ValueError('Only source-image group splits are supported')
    counts=Counter(); classes={s:Counter() for s in SPLITS}
    source_splits=defaultdict(set);seen_ids={};seen_images={};raw={}
    for r in m['records']:
        s=r['split'];assert s in SPLITS
        if r['id'] in seen_ids:raise ValueError('Duplicate crop ID')
        seen_ids[r['id']]=s
        if r['image_sha256'] in seen_images:raise ValueError('Duplicate crop image content')
        seen_images[r['image_sha256']]=s
        for key in ('image','mask'):
            if digest(ROOT/r[key])!=r[key+'_sha256']:raise ValueError('Crop asset changed')
        if r['raw_image'] not in raw:raw[r['raw_image']]=digest(ROOT/r['raw_image'])
        assert raw[r['raw_image']]==r['source_sha256']
        counts[s]+=1;classes[s].update(m['classes'][k] for k in r['labels'])
        source_splits[r['source_sha256']].add(s)
    if any(len(v)>1 for v in source_splits.values()):raise ValueError('Source image appears in multiple splits')
    n=len(m['records']);expected=[int(.7*n),int(.15*n),n-int(.7*n)-int(.15*n)]
    assert all(abs(counts[s]-expected[i])<=max(2,int(.02*n)) for i,s in enumerate(SPLITS))
    assert all(classes[s][c]>0 for s in SPLITS for c in m['classes'][1:])
    return {'manifest_sha256':digest(path),'dataset':m['dataset'],'split_unit':m['split_unit'],
            'samples':dict(counts),'class_presence':classes,'unique_sources':len(source_splits),
            'cross_split_crop_duplicates':0,
            'sources_in_multiple_splits':sum(len(v)>1 for v in source_splits.values()),
            'pairwise_source_overlap':{a+'_'+b:sum({a,b}<=v for v in source_splits.values())
                                      for a,b in [('train','val'),('train','test'),('val','test')]}}


class PaperDataset(SegmentationDataset):
    def __init__(self,manifest,split,size=512):
        m=json.loads(Path(manifest).read_text())
        self.records=[r for r in m['records'] if r['split']==split]
        self.size,self.training=size,split=='train'
        self.mean=torch.tensor([.485,.456,.406])[:,None,None]
        self.std=torch.tensor([.229,.224,.225])[:,None,None]


def make_loader(config,split,seed,batch_size=None,workers=None):
    path=ROOT/config['manifest'];m=json.loads(path.read_text())
    if m['classes']!=config['classes'] or digest(path)!=config['manifest_sha256']:
        raise ValueError('Manifest/config mismatch')
    data=PaperDataset(path,split,config['image_size']);batch=batch_size or config['training']['batch_size']
    assert split in SPLITS
    return DataLoader(data,batch_size=batch,shuffle=split=='train',
                      num_workers=config['training']['workers'] if workers is None else workers,
                      worker_init_fn=seed_worker,generator=torch.Generator().manual_seed(seed),
                      drop_last=split=='train' and len(data)%batch==1,pin_memory=torch.cuda.is_available())
