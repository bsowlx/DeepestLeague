"""RT-DETR Fine-tuning Script for Minimap Detection."""

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


def finetune(
    weights: str,
    dataset: str,
    run_id: str = None,
    run_name: str = None,
    config_path: str = None,
    **overrides,
):
    """Fine-tune RT-DETR model from pretrained weights."""

    if config_path:
        cfg = load_config(config_path)
    else:
        cfg = {
            "epochs": 100,
            "batch": 8,
            "imgsz": 256,
            "device": "0",
            "optimizer": "AdamW",
            "lr0": 0.00005,
            "lrf": 0.05,
            "weight_decay": 0.00005,
            "cos_lr": True,
            "warmup_epochs": 3,
            "patience": 20,
            "seed": 42,
            "deterministic": True,
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
            "close_mosaic": 0,
            "cutmix": 0,
            "copy_paste": 0,
            "auto_augment": None,
            "erasing": 0,
        }

    cfg.update(overrides)
    cfg.pop("pretrained", None)

    data_config = find_dataset(dataset)
    resolved_run_id = resolve_run_id(run_id, run_name, "finetune")
    output = ROOT / "results" / "experiments"
    rtdetr = RTDETR(weights)
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
            "mode": "finetune",
            "run_id": resolved_run_id,
            "weights": weights,
            "dataset": dataset,
            "data_config": str(data_config),
            "config_path": config_path,
            "overrides": overrides,
            "git": get_git_state(ROOT),
        },
    )

    return result


def main():
    parser = argparse.ArgumentParser(description="Fine-tune RT-DETR for minimap detection")
    parser.add_argument("--weights", required=True, help="Pretrained weights path")
    parser.add_argument("--dataset", required=True, help="Dataset folder name")
    parser.add_argument("--run_id", default=None, help="Standardized run ID")
    parser.add_argument("--run_name", default=None, help="Deprecated alias for run_id")
    parser.add_argument("--config", default=None, help="YAML config path")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--device", type=str)
    args = parser.parse_args()

    overrides = {}
    if args.epochs:
        overrides["epochs"] = args.epochs
    if args.batch:
        overrides["batch"] = args.batch
    if args.device:
        overrides["device"] = args.device

    finetune(
        weights=args.weights,
        dataset=args.dataset,
        run_id=args.run_id,
        run_name=args.run_name,
        config_path=args.config,
        **overrides,
    )


if __name__ == "__main__":
    main()
