"""Lossless tensor-name conversion of NVIDIA HF MiT-B2 ImageNet checkpoint."""
import argparse
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def digest(path):
 with path.open("rb") as stream:
  return hashlib.file_digest(stream, "sha256").hexdigest()

def convert(source, target, expected_source_sha, expected_output_sha):
 import torch
 if digest(source) != expected_source_sha:raise ValueError('MiT source SHA256 mismatch')
 if target.exists() or target.is_symlink():
  if digest(target) != expected_output_sha:raise ValueError('Refusing to replace unknown MiT output')
  return
 if torch.__version__.split('+')[0] != '2.5.1':raise RuntimeError('Pinned serialization requires torch 2.5.1')
 original=torch.load(source,map_location='cpu',weights_only=True);result={}
 for k,v in original.items():
  if k.startswith('classifier.'):
   result[k.replace('classifier.','head.')]=v;continue
  k=k.removeprefix('segformer.encoder.')
  m=re.fullmatch(r'patch_embeddings\.(\d+)\.(.*)',k)
  if m:
   dest=f'patch_embed{int(m[1])+1}.'+m[2].replace('layer_norm.','norm.');result[dest]=v;continue
  m=re.fullmatch(r'layer_norm\.(\d+)\.(.*)',k)
  if m:result[f'norm{int(m[1])+1}.{m[2]}']=v;continue
  m=re.fullmatch(r'block\.(\d+)\.(\d+)\.(.*)',k)
  if not m:raise ValueError(k)
  suffix=m[3]
  mapping={'layer_norm_1':'norm1','layer_norm_2':'norm2','attention.output.dense':'attn.proj',
   'attention.self.query':'attn.q','attention.self.key':'attn.key','attention.self.value':'attn.value',
   'attention.self.sr':'attn.sr','attention.self.layer_norm':'attn.norm','mlp.dense1':'mlp.fc1','mlp.dense2':'mlp.fc2','mlp.dwconv.dwconv':'mlp.dwconv.dwconv'}
  for a,b in mapping.items():
   if suffix.startswith(a+'.'):
    result[f'block{int(m[1])+1}.{m[2]}.'+b+suffix[len(a):]]=v;break
  else:raise ValueError(k)
 for k in list(result):
  if '.attn.key.' in k:
   value=k.replace('.key.','.value.');result[k.replace('.key.','.kv.')]=torch.cat([result.pop(k),result.pop(value)],0)
 target.parent.mkdir(parents=True, exist_ok=True)
 with tempfile.TemporaryDirectory(prefix='.mit-convert-', dir=target.parent) as directory:
  staged=Path(directory)/'mit_b2.pth'
  torch.save(result,staged)
  if digest(staged)!=expected_output_sha:
   raise ValueError('Converted MiT SHA256 mismatch; no output installed')
  os.link(staged,target)


def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--source',type=Path,default=ROOT/'modern_study/assets/mit_b2_hf.bin')
 parser.add_argument('--target',type=Path,default=ROOT/'modern_study/assets/mit_b2.pth')
 args=parser.parse_args()
 assets=json.loads((ROOT/'configs/assets.lock.json').read_text())['assets']
 convert(args.source,args.target,assets['mit_b2_source']['sha256'],assets['mit_b2']['sha256'])
 print('Verified MiT conversion:', args.target)


if __name__=='__main__':main()
