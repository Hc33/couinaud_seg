"""Explicit single-case inference for the confirmed three-stage model."""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from models import Attention_MT_Networks_with_L0
from data import load_case, load_pickle, save_predictions


MODEL_NAME = "Attention_MT_Networks_with_L0"
MODEL_ARGUMENTS = dict(in_ch=1, out_ch_1=3, out_ch_2=9, dropout=True)


def read_config(path):
    """Resolve every relative path against the configuration file directory."""
    path = Path(path).resolve()
    config = json.loads(path.read_text())
    for key in ("input", "properties", "metadata", "output"):
        config[key] = str((path.parent / config[key]).resolve())
    config["checkpoints"] = {
        key: str((path.parent / value).resolve())
        for key, value in config["checkpoints"].items()
    }
    return config


def load_model(config):
    if config["model"] != MODEL_NAME or config["model_kwargs"] != MODEL_ARGUMENTS:
        raise ValueError("Only Attention_MT_Networks_with_L0(1, 3, 9, dropout=True) is confirmed.")
    device = torch.device(config["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available. Set device to cpu.")
    checkpoint = config["checkpoints"][str(config["fold"])]
    model = Attention_MT_Networks_with_L0(**config["model_kwargs"])
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def predict(model, ct_nda):
    """Legacy _inference_one_case; only device placement is configurable."""
    device = next(model.parameters()).device
    with torch.no_grad():
        ct_nda = torch.FloatTensor(ct_nda).to(device)
        ct_nda = ct_nda.unsqueeze(0).unsqueeze(0)
        out1_tensor, out2_tensor = model(ct_nda)
        out1_tensor = F.softmax(out1_tensor, dim=1)
        out2_tensor = F.softmax(out2_tensor, dim=1)
    return out1_tensor, out2_tensor


def run(config):
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.set_num_threads(config["threads"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model = load_model(config)
    properties = load_pickle(config["properties"])
    metadata = load_pickle(config["metadata"])
    ct, vessel_label, couinaud_label = load_case(config["input"], properties)
    probabilities = predict(model, ct)
    paths = save_predictions(probabilities, metadata, config["output"],
                             config["save_probabilities"])
    print("Vessel:", tuple(probabilities[0].shape), "Couinaud:", tuple(probabilities[1].shape))
    print(json.dumps(paths, indent=2))
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=range(1, 6), help="Override the configured weight fold.")
    parser.add_argument("--output", type=Path, help="Override output directory (relative to current directory).")
    args = parser.parse_args()
    config = read_config(args.config)
    if args.fold is not None:
        config["fold"] = args.fold
    if args.output is not None:
        config["output"] = str(args.output.resolve())
    run(config)


if __name__ == "__main__":
    main()
