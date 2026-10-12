Historical implementation notes; current usage and outcomes are in the root README.

# Validation status

Local checks passed on Windows, Python 3.12.14, torch 2.5.1+cpu and
torchvision 0.20.1+cpu. The server uses Python 3.10 and CUDA 12.1 wheels;
repeat `python ald_check.py` there before training.

- Dynamic-convolution numerical equivalence and input-dependent routing.
- WIoU v1/v3 finite gradients, identical-box zero loss, validation mean freeze.
- Two DynConvLayer modules and three FANet modules in the complete YAML graph.
- Both dual-head branches and P3/P4/P5 strides 8/16/32.
- Synthetic foreground and empty-target loss; custom-module gradients.
- Trainable parameter coverage in the upstream SGD optimizer.
- Optimizer step, inference and model/loss-state checkpoint roundtrip.
- Installer clean install, idempotence and refusal of conflicting edits.

Full training graph parameters: 51,333,668 (includes auxiliary branch).
No dataset mAP, GPU throughput, AMP or real-data training has been validated yet.
Dataset audit/launcher syntax is checked; real dataset validation is pending.
