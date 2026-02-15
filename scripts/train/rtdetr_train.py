"""RT-DETR Training Script for Minimap Detection."""

import argparse
from pathlib import Path

import yaml
from ultralytics import RTDETR

from src.run_utils import get_git_state, resolve_run_id, resolve_ultralytics_output_dir, write_metadata

ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def find_dataset(name: str) -> Path:
    candidates = [
        ROOT / "results" / "configs" / f"{name}.yaml",
        ROOT / "data" / "synthetics" / name / "config.yaml",
        ROOT / "data" / "replays" / name / "config.yaml",
    ]
    for cfg in candidates:
        if cfg.exists():
            return cfg
    raise FileNotFoundError(f"Dataset '{name}' not found")


def resolve_model(model: str | None) -> str:
    if model:
        return model
    return str(ROOT / "configs" / "rtdetr" / "rtdetr-x.yaml")


def train(
    model: str,
    dataset: str,
    run_id: str = None,
    run_name: str = None,
    config_path: str = None,
    **overrides,
):
    """Train RT-DETR model on minimap dataset."""

    if config_path:
        cfg = load_config(config_path)
    else:
        cfg = {
            "epochs": 100,
            "batch": 64,
            "imgsz": 256,
            "device": "0",
            "optimizer": "AdamW",
            "lr0": 0.0002,
            "lrf": 0.05,
            "weight_decay": 0.0003,
            "cos_lr": True,
            "warmup_epochs": 10,
            "patience": 10,
            "seed": 42,
            "deterministic": True,
            "pretrained": False,
            "mosaic": 0,
            "mixup": 0,
            "hsv_h": 0.01,
            "hsv_s": 0.2,
            "hsv_v": 0.1,
            "degrees": 0,
            "translate": 0,
            "scale": 0,
            "shear": 0,
            "perspective": 0,
            "flipud": 0,
            "fliplr": 0,
            "bgr": 0,
            "mosaic": 0,
            "close_mosaic": 0,
            "mixup": 0,
            "cutmix": 0,
            "copy_paste": 0,
            "auto_augment": None,
            "erasing": 0,
        }

    cfg.update(overrides)

    data_config = find_dataset(dataset)
    model_path = resolve_model(model)
    resolved_run_id = resolve_run_id(run_id, run_name, "train")
    output = ROOT / "results" / "experiments"
    rtdetr = RTDETR(model_path)
    result = rtdetr.train(
        data=str(data_config),
        project=str(output),
        name=resolved_run_id,
        **cfg,
    )

    output_dir = resolve_ultralytics_output_dir(result, output / resolved_run_id)
    write_metadata(
        output_dir,
        {
            "mode": "train",
            "run_id": resolved_run_id,
            "model": model_path,
            "dataset": dataset,
            "data_config": str(data_config),
            "config_path": config_path,
            "overrides": overrides,
            "git": get_git_state(ROOT),
        },
    )

    return result


def main():
    parser = argparse.ArgumentParser(description="Train RT-DETR for minimap detection")
    parser.add_argument("--model", default=None, help="Model path (default: configs/rtdetr/rtdetr-x.yaml)")
    parser.add_argument("--dataset", required=True, help="Dataset folder name")
    parser.add_argument("--run_id", default=None, help="Standardized run ID")
    parser.add_argument("--run_name", default=None, help="Deprecated alias for run_id")
    parser.add_argument("--config", default=None, help="YAML config path")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--device", type=str)
    parser.add_argument("--save_json", action="store_true", help="Save predictions as COCO-style JSON")
    args = parser.parse_args()

    overrides = {}
    if args.epochs:
        overrides["epochs"] = args.epochs
    if args.batch:
        overrides["batch"] = args.batch
    if args.device:
        overrides["device"] = args.device
    if args.save_json:
        overrides["save_json"] = True

    train(
        model=args.model,
        dataset=args.dataset,
        run_id=args.run_id,
        run_name=args.run_name,
        config_path=args.config,
        **overrides,
    )


if __name__ == "__main__":
    main()
