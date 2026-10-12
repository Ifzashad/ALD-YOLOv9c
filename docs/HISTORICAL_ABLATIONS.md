Historical implementation notes; current usage and outcomes are in the root README.

# Controlled ALD-YOLOv9c reconstruction experiments

Each experiments/*.yaml is an executable experiment specification. Baseline, DynConv-only,
FA-Net-only and W-IoU-v3-only train independently from the original COCO yolov9-c.pt.
Full uses all three components. Never initialize ablations from the trained full model.
All use the unchanged Roboflow v1 splits, 640 pixels, batch 8, SGD, seed 0,
200 maximum epochs, patience 50 and close-mosaic 15. The hyperparameter files
are identical except for the selected box loss. CIoU is used without W-IoU.
Dual detection / auxiliary PGI remains in every variant.

DynConv changes backbone stem layers 1 and 2 only. FA-Net adds three modules on
main P3/P4/P5 before detection; it leaves auxiliary features unchanged.
Configs describe a documented reconstruction, not recovered original source.
See ALD_README.md for paper ambiguities and the dataset discrepancy.

Run ald_ablation_check.py first, then ald_ablation_queue.py --gpu 0.
The queue trains baseline and the three single-component variants sequentially.
It does not retrain the completed full model. Each run uses validation for early
stopping and evaluates its best.pt once on test. Early stopping does not prove
absence of overfitting or underfitting. Never select hyperparameters on test.
The upstream validation implementation does not compute validation loss;
zero validation-loss CSV columns must not be plotted as measured losses.
Use mAP and precision/recall. warn_only deterministic mode permits nondeterministic
CUDA operations; one seed is not evidence of statistical significance.

Each queue output contains configurations, commands, hashes, dependency versions,
logs and test metrics. Failures stop the queue. It never overwrites an old run.
For repeated seeds, run a separately declared study; do not mix seeds in this table.
The full model must be evaluated with the same test arguments before combining
scores if its original evaluation arguments cannot be established.

## Existing full results (rounded, transcribed from supplied logs)
Validation: 968 images, P .846, R .780, mAP50 .832, mAP50-95 .549.
Test: 501 images / 966 objects, P .737, R .750, mAP50 .763, mAP50-95 .495.
Early stopping: 133 epochs completed; reported best epoch 82; patience 50.
Checkpoint: <original-workspace>/runs/ald-full-20261010-015544/weights/best.pt.
No baseline or single-component scores are available yet. Do not invent them.

## External generalization (pending)
Candidate: SeaClear, https://github.com/adjuras/seaclear-dataset
Dataset DOI: https://doi.org/10.4121/4f1dff25-e157-4399-a5d4-478055461689.v1
Before evaluation, inspect its actual annotation categories, document an explicit
mapping to compatible training classes, and check source/near-duplicate overlap.
Do not assume all 15 classes correspond. Define the evaluated class subset,
unmapped-object handling, image inclusion and prediction filtering before seeing
scores. Keep original annotations and publish a mapping plus conversion script.
Freeze the original best checkpoint: no external fine-tuning, early stopping,
threshold selection or cherry-picking. Evaluate baseline and full under the same
protocol to support comparative generalization. Report image/object counts,
per-class AP, global metrics and failure examples. External evaluation has not run.

GitHub can contain only external metrics/plots plus reproducible evaluation code,
mapping, dataset version/source and citations. Raw external images need not be
committed. Keep environments, caches, datasets, secrets and checkpoints out of
ordinary source commits; distribute trained weights separately with hashes.
