"""Portable entry point for training and held-out evaluation. Run from any cwd."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser()
    p.add_argument('task',choices=['train','evaluate'])
    p.add_argument('--variant',choices=['baseline','dyn-only','fa-only','wiou-only','full'],default='full')
    p.add_argument('--device',default='0',help='Actual CUDA device index, or cpu for evaluation')
    p.add_argument('--weights',type=Path)
    p.add_argument('--split',choices=['val','test'],default='test')
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--epochs',type=int,default=200)
    p.add_argument('--dry-run',action='store_true')
    args=p.parse_args()
    if args.batch_size<1 or args.epochs<1: p.error('Batch size and epochs must be positive')
    os.chdir(ROOT);sys.path.insert(0,str(ROOT))
    for key,suffix in [('YOLOV5_CONFIG_DIR','.cache/ultralytics'),('TORCH_HOME','.cache/torch'),
                       ('XDG_CACHE_HOME','.cache'),('MPLCONFIGDIR','.cache/matplotlib'),('TMPDIR','.cache/tmp')]:
        os.environ[key]=str(ROOT/suffix);(ROOT/suffix).mkdir(parents=True,exist_ok=True)
    os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
    import matplotlib
    font=Path(matplotlib.get_data_path())/'fonts/ttf/DejaVuSans.ttf'
    for name in ('Arial.ttf','Arial.Unicode.ttf'):
        dest=ROOT/'.cache/ultralytics'/name
        if not dest.exists():shutil.copy2(font,dest)
    import yaml
    # YOLO resolves paths against its code root; a generated absolute path works
    # regardless of the checkout location. This generated file is not committed.
    data=yaml.safe_load((ROOT/'datasets/AD2025/data.yaml').read_text())
    data['path']=str(ROOT/'datasets/AD2025')
    local_data=ROOT/'.cache/ad2025-local.yaml';local_data.write_text(yaml.safe_dump(data,sort_keys=False))
    weights=args.weights or Path('weights/yolov9-c.pt' if args.task=='train' else 'weights/best.pt')
    weights=weights if weights.is_absolute() else ROOT/weights
    if not weights.is_file():p.error('Missing weights. Run git lfs pull: '+str(weights))
    if weights.stat().st_size<1024:p.error('Weights look like an LFS pointer; run git lfs pull')
    stamp=datetime.now().strftime('%Y%m%d-%H%M%S')
    name=f'{args.variant}-{args.task}-{stamp}'
    if args.task=='train':
        cmd=[sys.executable,'train_dual.py','--cfg',f'models/ablations/{args.variant}.yaml',
             '--hyp',f'data/hyps/ablations/{args.variant}.yaml','--data',str(local_data),
             '--weights',str(weights),'--img','640','--batch-size',str(args.batch_size),
             '--epochs',str(args.epochs),'--optimizer','SGD','--patience','50',
             '--close-mosaic',str(min(15,args.epochs-1)),'--workers','4','--seed','0',
             '--device',args.device,'--project',str(ROOT/'runs'),'--name',name]
        print('Command:',cmd,flush=True)
        if not args.dry_run:subprocess.run(cmd,check=True)
    else:
        if args.dry_run:
            print('Evaluation:',weights,'split',args.split,'device',args.device);return
        from val_dual import run
        metrics,_,timing=run(data=str(local_data),weights=str(weights),task=args.split,
            device=args.device,batch_size=args.batch_size,imgsz=640,workers=4,half=False,
            conf_thres=.001,iou_thres=.7,max_det=300,augment=False,project=str(ROOT/'runs'),name=name)
        result=dict(zip(('precision','recall','map50','map50_95'),map(float,metrics[:4])))
        result.update(split=args.split,timing_ms=list(timing))
        (ROOT/'runs'/name/'metrics.json').write_text(json.dumps(result,indent=2))
        print(json.dumps(result,indent=2))

if __name__=='__main__':main()
