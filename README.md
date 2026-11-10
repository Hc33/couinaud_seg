# Couinaud model and inference

Three-stage model with attention sharing and task-specific IR, DWB weight calculation, and single-case inference. This source package does not include training scripts, pretrained weights, datasets, or a raw-CT preprocessing pipeline.

## Setup

The server reference environment used Python 3.8.20, PyTorch 2.4.1, NumPy 1.24.3 and SimpleITK 2.2.1 on CPU. From this directory, install the pinned dependencies:

```bash
python -m pip install -r requirements.txt
```

## Model example

Run this Python example from the package directory. It uses random weights to check output shapes; its outputs are not trained predictions.

```python
import torch
from models import Attention_MT_Networks_with_L0

model = Attention_MT_Networks_with_L0(1, 3, 9, dropout=True).eval()
with torch.no_grad():
    vessel, couinaud = model(torch.zeros(1, 1, 32, 32, 32))
print(tuple(vessel.shape), tuple(couinaud.shape))
# (1, 3, 32, 32, 32) (1, 9, 32, 32, 32)
```

## Inference with existing preprocessed data

Replace the example paths in `inference.json` with your files, then run:

```bash
python inference.py --config inference.json
```

Paths resolve relative to the JSON file. The checkpoint must be a plain model `state_dict` matching the configured three-stage model. Loading is strict. `fold` selects the corresponding checkpoint entry; only that checkpoint is needed for a run.

The input contract follows the supplied preprocessing workflow:

- NPZ: key `data`, shape `[channel, Z, Y, X]`. Index 0 contains CT, index 2 vessel labels, and index 3 Couinaud labels. The loader reads both labels; prediction uses CT only. The reference case used a `128 x 128 x 128` grid.
- Properties pickle: `liver_intensity_mean`, `liver_intensity_std`, `liver_intensity_00_5`, and `liver_intensity_99_5`.
- Geometry pickle: three NumPy arrays in Z,Y,X order named `cropped_size`, `size_after_resample`, and `original_size_of_raw_data`; the six inclusive crop bounds `before_crop_nonzero_region_{z,y,x}_{min,max}`; and the SimpleITK geometry fields `itk_direction`, `itk_orgin`, and `itk_spacing`. The spelling `itk_orgin` is retained from the input format.

Outputs are `vessel.nii.gz` and `couinaud.nii.gz` in the restored image geometry. `save_probabilities` also writes model-grid softmax arrays to `probabilities.npz`. The retained export path thresholds each channel at 0.5 before interpolation, crop restoration and final argmax. Label ranges are 0-2 and 0-8, including background 0.

`dwb.py` provides the two-loss weighting formula without assigning loss classes or a training schedule. The historical full training configuration has not been recovered. Training and CUDA execution have not been validated for this package.
