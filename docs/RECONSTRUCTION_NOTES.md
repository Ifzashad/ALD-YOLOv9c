Historical implementation notes; current usage and outcomes are in the root README.

# ALD-YOLOv9c reconstruction

This project reconstructs the supplied ALD-YOLOv9c manuscript after the original
source and trained weights were lost. It is not the recovered original code.
No new dataset accuracy, paper-result reproduction or GPU speed is claimed yet.

## Architecture

Base: WongKinYiu/yolov9 commit `5b1ea9a8b3f0ffe4fe0e203ec6232d788bb3fcff`.
Figure 6 is the authority for placement. Layers 1 and 2 become DynConvLayer;
all GELAN, ADown, SPPELAN, neck concatenations and PGI auxiliary routing remain.
Main P3/P4/P5 outputs (16, 19, 22) feed FANet layers 38/39/40, then the main
half of DualDDetect at 41. Auxiliary inputs remain 31/34/37. FANet does not
feed back into neck downsampling, matching the figure's branching connections.

DynConvLayer follows Figure 7: GAP -> MLP -> softmax -> weighted kernel mixture
-> convolution -> BN -> SiLU. Four experts, router reduction 4 (minimum hidden
width 4), and SiLU are explicit choices because the paper omits these settings.
FANet follows Figure 9/Eqs. 3-7: shared avg/max channel MLP, reduction 16, followed
by pooled-channel concatenation and summed 3/5/7 spatial convolutions + sigmoid.
The ratio 16 and bias-free attention convolutions are reconstruction choices.

## Wise-IoU

`hyp.ald.yaml` selects WIoU-v3, interpreting the paper's non-monotonic focusing
description. `hyp.ald-v1.yaml` selects distance-attention-only WIoU-v1, matching
its displayed equations 8-10. The manuscript does not resolve this ambiguity.
v3 uses alpha=1.9, delta=3, EMA half-life=7000 positive-batch updates, initial
mean=1. These are documented choices. Both dual branches use WIoU; classification,
DFL and task-aligned assignment stay upstream. Validation does not update the EMA.
Training checkpoints save/restore both branches' loss state. The supported
launcher uses one allocated GPU; distributed WIoU-statistic synchronization
has not been implemented or validated.

## Dataset and paper limitations

The author confirmed use of Roboflow version 1 without modification:
https://universe.roboflow.com/faiza-rehman/underwater-garbage-detection-d4dnn/dataset/1
The published version reports 5,095 images: 3,626 train / 968 valid / 501 test,
15 classes. Preserve this export's actual class-ID order and split. A data YAML
must be generated from the downloaded export, not from the paper's display order.
The manuscript's AD-2025 merged-data provenance, 5,070 count and 70/20/10 split
do not describe this confirmed export. Document the correction publicly.

The diagram and multi-kernel equations do not specify a mechanism that achieves
the paper's reduced parameter count. This implementation does not invent one:
independent experts and attention add parameters. Do not claim 23.5M parameters,
101 GFLOPs, an 8.8-point gain, or any FPS until measured on this reconstruction.
Cross-layer sharing is not implemented: the two stem convolutions have different
tensor shapes and the paper supplies no sharing/projection specification.
The full dual-branch training graph includes auxiliary parameters and is not
directly comparable to an inference-only parameter count. Upstream THOP profiling
does not account for this custom functional dynamic convolution: its printed
FLOPs must not be reported as a measured ALD FLOPs result.

## Training settings

Paper values: 640 input, batch 8, max 200 epochs, SGD lr 0.001, momentum 0.931,
weight decay 0.0005, patience 50, mosaic 1.0, close mosaic last 15 epochs.
Remaining scheduler/warmup/HSV/scale/flip values follow upstream scratch-high
defaults. MixUp and copy-paste are disabled. Cutout is not added because the
paper provides no probability/magnitude; this is a known reconstruction gap.
Upstream gradient accumulation nominal batch 64 is retained. Stopping at epoch
185 is not forced. Batch changes or a reduced smoke run must be reported.

## Install and check

Use the supplied `install_ald.py` with the existing unmodified pinned checkout.
It verifies original file hashes, checks all targets before changing anything,
keeps originals in `.ald-backup`, and refuses conflicting edits. Reruns are safe.
No network access, package installation or training is performed by the installer.

```bash
conda activate <original-workspace>/env
python <original-workspace>/install_ald.py --repo <original-workspace>/code/ald-yolov9c
cd <original-workspace>/code/ald-yolov9c
python ald_check.py
```

After dataset inspection and GPU allocation, use `python ald_train.py --help`.
Supply explicit local dataset/checkpoint paths and the allocated physical GPU ID.
First use `--dry-run`, then a two-epoch run before full training. Baseline and ALD
must use the same data split and training protocol. Save `pip freeze` with results.
Source pretrained checkpoints must be trusted: upstream uses Python pickle.
Pretrained transfer maps head 38 to 41 and initializes dynamic experts from
their corresponding original Conv. Changed class-head tensors are reinitialized.

`ald_check.py` checks numerical equivalence of dynamic convolution to an explicit
expert sum, input-dependent routing, WIoU stability/gradients/state roundtrip,
model routing, both heads, synthetic foreground and empty-target losses, custom
gradients, optimizer coverage, inference and serialization. It is not a dataset
training/evaluation test. GPU AMP and real data remain to be checked on the server.

## GitHub and attribution

Keep the upstream GPL-3.0 license and attribution. Custom code is GPL-3.0-only.
Keep data, environments, credentials and weights out of Git. Publish verified
weights separately with the training recipe, class names, split manifest and
measured metrics. Preserve this reconstruction note when writing the final README.

References:
- YOLOv9: https://github.com/WongKinYiu/yolov9
- Wise-IoU: https://arxiv.org/abs/2301.10051
- Wise-IoU reference: https://github.com/Instinct323/Wise-IoU
- Supplied manuscript: Figures 6, 7, 9; Eqs. 1-10; Tables 2-3.
