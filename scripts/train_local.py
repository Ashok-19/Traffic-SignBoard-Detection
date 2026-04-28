#!/usr/bin/env python3
"""Train YOLO26 on the merged 8-class signboard dataset (dataset1 + dataset2 + dataset5)."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO


def detect_device(requested_device: str) -> str:
    """Auto-detect best device. Fall back to CPU if CUDA unavailable."""
    if requested_device == "0" or requested_device.startswith("cuda"):
        if not torch.cuda.is_available():
            print("[warn] CUDA not available; falling back to CPU")
            return "cpu"
    return requested_device


def main() -> int:
    parser = argparse.ArgumentParser(description="Train YOLO26 on the merged 8-class signboard dataset.")
    parser.add_argument("--data", default="signboard-yolo/merged_dataset/data.yaml")
    parser.add_argument("--model", default="yolo26n.pt", help="Lightweight YOLO26 base model.")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="0", help="Device to use. Set to 'cpu' for CPU-only or '0' for GPU. Auto-falls back to CPU if CUDA unavailable.")
    parser.add_argument("--project", default="runs/signboard")
    parser.add_argument("--name", default="yolo26n_signboard_merged_8cls")
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--close-mosaic", type=int, default=15)
    parser.add_argument("--export", default="onnx", help="Export format after training: onnx, torchscript, openvino, none.")
    args = parser.parse_args()

    data_path = Path(args.data).resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {data_path}")

    device = detect_device(args.device)
    model = YOLO(args.model)
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=device,
        project=args.project,
        name=args.name,
        seed=args.seed,
        patience=args.patience,
        close_mosaic=args.close_mosaic,
        pretrained=True,
        cos_lr=True,
        hsv_h=0.02,
        hsv_s=0.5,
        hsv_v=0.4,
        degrees=7.0,
        translate=0.1,
        scale=0.6,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
    )

    best = Path(model.trainer.best) if getattr(model, "trainer", None) and model.trainer.best else None
    if args.export.lower() != "none" and best and best.exists():
        YOLO(str(best)).export(format=args.export, imgsz=args.imgsz)
        print(f"[done] exported {args.export} model from {best}")
    else:
        print("[done] training complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
