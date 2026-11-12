# Attention-based co-learning for liver segmentation

Three-stage model with attention sharing and task-specific IR, DWB weight calculation, and single-case inference. This source package does not include training scripts, pretrained weights, datasets, or a raw-CT preprocessing pipeline.

## Paper components

| Paper component | Implementation |
| --- | --- |
| Joint vessel and Couinaud segmentation network | `AttentionCoLearningNet` in `models/model.py` |
| Sharing module | `AttentionSharingBlock` in `models/model.py` |
| Attention-based Co-learning Block (ACB) | Feature fusion, `ChannelAttention`, and `SpatialAttention` within the sharing module |
| Information Routing (IR) | Two task-specific `InformationRoutingBlock` instances per sharing module, each using `L0Conv3d` in `models/ir.py` |
| Decoder feature fusion | `DecoderFusionBlock` in `models/layers.py` |
| Dynamic Weight Balancing (DWB) | `compute_dwb_weights` in `dwb.py` |

At each of the three decoder stages, `AttentionSharingBlock` concatenates the vessel and Couinaud features. Feature fusion uses a 3 x 3 x 3 convolution, batch normalization, and ReLU. `ChannelAttention` reweights the fused channels. `SpatialAttention` concatenates the channel-wise maximum and mean maps, then applies a 5 x 5 x 5 convolution, batch normalization, and sigmoid to obtain spatial weights.

ACB supplies the same attended feature to the two task-specific IR blocks at that stage. The network contains six IR blocks in total. Each IR block adds its gated convolutional path to its input. The decoder multiplies the routed feature by its own feature, applies convolution, and adds its residual feature.

The custom `L0Conv3d` implementation is retained. With the default `local_rep=False`, training gates convolutional input channels through weight multiplication. Inference gates convolutional output channels. The naming update preserves both branches.

## Model names and checkpoint compatibility

Import `AttentionCoLearningNet` from `models`. Use `vessel_classes` and `couinaud_classes` in `model_kwargs`, as shown in `inference.json`. The inference entry point supports `in_ch=1`, `vessel_classes=3`, `couinaud_classes=9`, and `dropout=True`.

Registered module and parameter names are unchanged, so existing plain `state_dict` keys remain compatible with the same model configuration. The public class names have changed. Old checkpoints serialized as complete Python model objects are not supported by this entry point.

## Setup

The server reference environment for the original package used Python 3.8.20, PyTorch 2.4.1, NumPy 1.24.3, and SimpleITK 2.2.1 on CPU. From this directory, install the pinned dependencies:

```bash
python -m pip install -r requirements.txt
```

## Model example

Run this Python example from the package directory. It uses random weights to check output shapes; its outputs are not trained predictions.

```python
import torch
from models import AttentionCoLearningNet

model = AttentionCoLearningNet(in_ch=1, vessel_classes=3, couinaud_classes=9, dropout=True).eval()
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

Paths in `inference.json` resolve relative to that file. The `--output` override resolves relative to the current working directory. The checkpoint must be a plain model `state_dict` matching the configured three-stage model. Loading is strict. `fold` selects the corresponding checkpoint entry, and only that checkpoint is needed for a run.

The input contract follows the supplied preprocessing workflow:

- NPZ: key `data`, shape `[channel, Z, Y, X]`. Index 0 contains CT, index 2 vessel labels, and index 3 Couinaud labels. The loader reads both labels; prediction uses CT only. The reference case used a `128 x 128 x 128` grid.
- Properties pickle: `liver_intensity_mean`, `liver_intensity_std`, `liver_intensity_00_5`, and `liver_intensity_99_5`.
- Geometry pickle: three NumPy arrays in Z,Y,X order named `cropped_size`, `size_after_resample`, and `original_size_of_raw_data`; the six inclusive crop bounds `before_crop_nonzero_region_{z,y,x}_{min,max}`; and the SimpleITK geometry fields `itk_direction`, `itk_orgin`, and `itk_spacing`. The spelling `itk_orgin` is retained from the input format.

Outputs are `vessel.nii.gz` and `couinaud.nii.gz` in the restored image geometry. `save_probabilities` also writes model-grid softmax arrays to `probabilities.npz`. The retained export path thresholds each channel at 0.5 before interpolation, crop restoration and final argmax. Label ranges are 0-2 and 0-8, including background 0.

## Validation scope

The naming update has passed static equivalence checks against the original package. The renamed package has not yet been run with PyTorch or the original checkpoints. The example above specifies the expected shapes, not a completed runtime test of this version.

`dwb.py` provides `compute_dwb_weights` for the two-loss weighting formula without assigning loss classes or a training schedule. The historical full training configuration has not been recovered. Training and CUDA execution have not been validated for this package.
