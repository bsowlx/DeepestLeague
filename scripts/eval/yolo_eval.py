"""YOLO Evaluation Script for Minimap Detection."""

import argparse
from pathlib import Path

from ultralytics import YOLO

from src.run_utils import get_git_state, resolve_run_id, resolve_ultralytics_output_dir, write_metadata

ROOT = Path(__file__).resolve().parents[2]


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


def evaluate(
    weights: str,
    dataset: str,
    eval_id: str = None,
    device: str = "0",
    split: str = "test",
    imgsz: int | None = None,
    save_json: bool = False,
):
    """Evaluate YOLO model and return metrics."""

    data_config = find_dataset(dataset)
    model = YOLO(weights)

    resolved_eval_id = resolve_run_id(eval_id, None, "eval")
    project = ROOT / "runs" / "detect"
    metrics = model.val(
        data=str(data_config),
        split=split,
        device=device,
        imgsz=imgsz,
        project=str(project),
        name=resolved_eval_id,
        save_json=save_json,
    )

    output_dir = resolve_ultralytics_output_dir(metrics, project / resolved_eval_id)
    write_metadata(
        output_dir,
        {
            "mode": "eval",
            "eval_id": resolved_eval_id,
            "weights": weights,
            "dataset": dataset,
            "data_config": str(data_config),
            "split": split,
            "device": device,
            "save_json": save_json,
            "git": get_git_state(ROOT),
        },
    )

    results = {
        "mAP50": metrics.box.map50,
        "mAP50-95": metrics.box.map,
        "precision": metrics.box.mp,
        "recall": metrics.box.mr,
    }

    speed = metrics.speed
    total_ms = speed.get("preprocess", 0) + speed.get("inference", 0) + speed.get("postprocess", 0)
    results["fps"] = 1000.0 / total_ms if total_ms > 0 else 0

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate YOLO model")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--eval_id", default=None, help="Standardized eval ID")
    parser.add_argument("--device", default="0")
    parser.add_argument("--split", default="test")
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--save_json", action="store_true")
    args = parser.parse_args()

    results = evaluate(
        args.weights,
        args.dataset,
        eval_id=args.eval_id,
        device=args.device,
        split=args.split,
        imgsz=args.imgsz,
        save_json=args.save_json,
    )

    print("\nResults:")
    for key, value in results.items():
        print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    main()
