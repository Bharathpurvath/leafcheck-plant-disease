"""Create stable, class-stratified train/validation and independent test folders."""
from pathlib import Path
import argparse, json, os, random
from collections import Counter

IMAGE_SUFFIXES={'.jpg','.jpeg','.png'}

def main():
 p=argparse.ArgumentParser(); p.add_argument('--data',required=True,help='PlantVillage raw/color folder'); p.add_argument('--output',default='data/split'); p.add_argument('--max-trainval-per-class',type=int,default=250); p.add_argument('--test-fraction',type=float,default=.2); p.add_argument('--seed',type=int,default=42); a=p.parse_args()
 src=Path(a.data); dst=Path(a.output); rng=random.Random(a.seed); report={}
 classes=sorted(d.name for d in src.iterdir() if d.is_dir())
 if len(classes)!=38: raise SystemExit(f'Expected 38 class folders; found {len(classes)} in {src}')
 for cls in classes:
  files=sorted(f for f in (src/cls).iterdir() if f.suffix.lower() in IMAGE_SUFFIXES)
  if len(files)<2: raise SystemExit(f'Insufficient images in {cls.name}')
  rng.shuffle(files); ntest=max(1,round(len(files)*a.test_fraction)); test=files[:ntest]; trainval=files[ntest:]
  rng.shuffle(trainval); trainval=trainval[:a.max_trainval_per_class]
  for split, group in [('test',test),('trainval',trainval)]:
   target=dst/split/cls; target.mkdir(parents=True,exist_ok=True)
   for f in group:
    dest=target/f.name
    if not dest.exists():
     try: os.link(f,dest)
     except OSError: dest.write_bytes(f.read_bytes())
  report[cls]={'available':len(files),'trainval':len(trainval),'test':len(test)}
 summary={'source':str(src.resolve()),'seed':a.seed,'max_trainval_per_class':a.max_trainval_per_class,'test_fraction':a.test_fraction,'class_count':len(classes),'class_counts':report,'trainval_total':sum(x['trainval'] for x in report.values()),'test_total':sum(x['test'] for x in report.values())}
 dst.mkdir(parents=True,exist_ok=True); (dst/'split_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8'); print(json.dumps({k:v for k,v in summary.items() if k not in ('class_counts','source')},indent=2))
if __name__=='__main__':main()


