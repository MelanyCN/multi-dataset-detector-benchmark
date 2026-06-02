import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import psutil
import pynvml
import supervision as sv
import torch
import yaml
from PIL import Image
from rfdetr import RFDETRMedium
from rfdetr.assets.coco_classes import COCO_CLASSES


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Run RF-DETR benchmark inference.")
    parser.add_argument(
        "--config",
        default="configs/rfdetr_config.yaml",
        help="Path to the YAML configuration file.",
    )
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def project_root_from_config(config):
    return Path(config.get("project_root", ".")).expanduser().resolve()


def resolve_path(path_value, project_root):
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return project_root / path


def init_gpu_handle():
    if not torch.cuda.is_available():
        return None

    try:
        pynvml.nvmlInit()
        return pynvml.nvmlDeviceGetHandleByIndex(0)
    except pynvml.NVMLError:
        return None


def gpu_mem_mb(gpu_handle):
    if gpu_handle is None:
        return 0

    info = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
    return info.used / 1024 / 1024


def gpu_util_percent(gpu_handle):
    if gpu_handle is None:
        return 0

    return pynvml.nvmlDeviceGetUtilizationRates(gpu_handle).gpu


def ram_mb():
    return psutil.Process().memory_info().rss / 1024 / 1024


def dataset_id_from_name(name):
    return name.split("_")[0]


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def detection_to_rows(image_name, dataset_id, detections):
    rows = []

    for i, box in enumerate(detections.xyxy):
        x1, y1, x2, y2 = box.tolist()
        class_id = int(detections.class_id[i])
        conf = float(detections.confidence[i])
        class_name = COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else str(class_id)

        rows.append(
            {
                "image_name": image_name,
                "dataset_id": dataset_id,
                "bbox_id": i,
                "class_id": class_id,
                "class_name": class_name,
                "confidence": conf,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "bbox_width": x2 - x1,
                "bbox_height": y2 - y1,
                "bbox_area": (x2 - x1) * (y2 - y1),
            }
        )

    return rows


def write_csv(path, rows):
    if not rows:
        return

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def average(values):
    return sum(values) / len(values) if values else 0


def main():
    args = parse_args()
    config = load_config(args.config)
    project_root = project_root_from_config(config)
    benchmark_config = config.get("rfdetr", {})

    model_name = benchmark_config.get("model_name", "rfdetr_medium_coco")
    confidence_threshold = float(benchmark_config.get("confidence_threshold", 0.25))
    warmup_iters = int(benchmark_config.get("warmup_iters", 30))

    input_dir = resolve_path(config["output_dataset"], project_root) / "images"
    out_dir = resolve_path(config.get("results_dir", "results"), project_root) / "rfdetr"
    out_img_dir = out_dir / "images_bbox"
    out_img_dir.mkdir(parents=True, exist_ok=True)

    pred_csv = out_dir / "predictions.csv"
    image_metrics_csv = out_dir / "image_metrics.csv"
    warmup_csv = out_dir / "warmup_metrics.csv"
    summary_json = out_dir / "summary_metrics.json"
    log_txt = out_dir / "run_log.txt"

    images = sorted([p for p in input_dir.iterdir() if p.suffix.lower() in IMG_EXTS])
    if not images:
        raise ValueError(f"No se encontraron imagenes en: {input_dir}")

    gpu_handle = init_gpu_handle()

    with open(log_txt, "w") as log:
        log.write(f"model={model_name}\n")
        log.write(f"confidence_threshold={confidence_threshold}\n")
        log.write(f"num_images={len(images)}\n")
        log.write(f"warmup_iters={warmup_iters}\n")
        log.write(f"torch_cuda={torch.cuda.is_available()}\n")
        if torch.cuda.is_available():
            log.write(f"gpu_name={torch.cuda.get_device_name(0)}\n")

    print("Cargando modelo RF-DETR...")
    if model_name != "rfdetr_medium_coco":
        raise ValueError(f"Modelo RF-DETR no soportado por este script: {model_name}")
    model = RFDETRMedium()

    print("Modelo cargado.")
    print(f"Imagenes encontradas: {len(images)}")

    warmup_rows = []
    warmup_images = images[: min(warmup_iters, len(images))]

    print("Iniciando warm-up...")
    for i in range(warmup_iters):
        img_path = warmup_images[i % len(warmup_images)]

        cpu_before = psutil.cpu_percent(interval=None)
        ram_before = ram_mb()
        gpu_mem_before = gpu_mem_mb(gpu_handle)
        gpu_util_before = gpu_util_percent(gpu_handle)

        sync()
        t0 = time.perf_counter()

        warmup_image = np.array(Image.open(img_path).convert("RGB"))
        _ = model.predict(warmup_image, threshold=confidence_threshold)

        sync()
        t1 = time.perf_counter()

        warmup_rows.append(
            {
                "warmup_iter": i + 1,
                "image_name": img_path.name,
                "warmup_time_ms": (t1 - t0) * 1000,
                "cpu_percent": psutil.cpu_percent(interval=None),
                "ram_mb": ram_mb(),
                "gpu_mem_mb": gpu_mem_mb(gpu_handle),
                "gpu_util_percent": gpu_util_percent(gpu_handle),
                "ram_before_mb": ram_before,
                "gpu_mem_before_mb": gpu_mem_before,
                "gpu_util_before_percent": gpu_util_before,
                "cpu_before_percent": cpu_before,
            }
        )

    print("Warm-up terminado.")

    prediction_rows = []
    image_metric_rows = []
    class_counter = Counter()
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    print("Iniciando inferencia medida...")
    for idx, img_path in enumerate(images, start=1):
        image_name = img_path.name
        dataset_id = dataset_id_from_name(image_name)
        pil_image = Image.open(img_path).convert("RGB")
        width, height = pil_image.size

        cpu_before = psutil.cpu_percent(interval=None)
        ram_before = ram_mb()
        gpu_mem_before = gpu_mem_mb(gpu_handle)
        gpu_util_before = gpu_util_percent(gpu_handle)

        sync()
        t0 = time.perf_counter()

        rgb_np = np.array(pil_image)
        detections = model.predict(rgb_np, threshold=confidence_threshold)

        sync()
        t1 = time.perf_counter()

        infer_ms = (t1 - t0) * 1000
        fps = 1000 / infer_ms if infer_ms > 0 else 0
        rows = detection_to_rows(image_name, dataset_id, detections)
        prediction_rows.extend(rows)

        for row in rows:
            class_counter[row["class_name"]] += 1

        labels = [
            f"pred_rfdetr: {row['class_name']} {row['confidence']:.2f}"
            for row in rows
        ]
        source_image = detections.metadata.get("source_image", pil_image)
        annotated = box_annotator.annotate(source_image, detections)
        annotated = label_annotator.annotate(annotated, detections, labels)
        Image.fromarray(annotated).save(out_img_dir / image_name)

        image_metric_rows.append(
            {
                "image_name": image_name,
                "dataset_id": dataset_id,
                "width": width,
                "height": height,
                "num_detections": len(rows),
                "inference_time_ms": infer_ms,
                "fps": fps,
                "cpu_percent": psutil.cpu_percent(interval=None),
                "ram_mb": ram_mb(),
                "gpu_mem_mb": gpu_mem_mb(gpu_handle),
                "gpu_util_percent": gpu_util_percent(gpu_handle),
                "cpu_before_percent": cpu_before,
                "ram_before_mb": ram_before,
                "gpu_mem_before_mb": gpu_mem_before,
                "gpu_util_before_percent": gpu_util_before,
            }
        )

        print(f"[{idx}/{len(images)}] {image_name} | det={len(rows)} | {infer_ms:.2f} ms")

    write_csv(warmup_csv, warmup_rows)
    write_csv(pred_csv, prediction_rows)
    write_csv(image_metrics_csv, image_metric_rows)

    infer_times = [r["inference_time_ms"] for r in image_metric_rows]
    fps_values = [r["fps"] for r in image_metric_rows]
    ram_values = [r["ram_mb"] for r in image_metric_rows]
    gpu_mem_values = [r["gpu_mem_mb"] for r in image_metric_rows]
    cpu_values = [r["cpu_percent"] for r in image_metric_rows]
    gpu_util_values = [r["gpu_util_percent"] for r in image_metric_rows]
    warmup_times = [r["warmup_time_ms"] for r in warmup_rows]

    summary = {
        "model_name": model_name,
        "weights": "official pretrained COCO",
        "confidence_threshold": confidence_threshold,
        "num_images": len(images),
        "warmup_iterations": warmup_iters,
        "avg_warmup_time_ms": average(warmup_times),
        "avg_inference_time_ms": average(infer_times),
        "min_inference_time_ms": min(infer_times),
        "max_inference_time_ms": max(infer_times),
        "avg_fps": average(fps_values),
        "avg_ram_mb": average(ram_values),
        "max_ram_mb": max(ram_values),
        "avg_gpu_mem_mb": average(gpu_mem_values),
        "max_gpu_mem_mb": max(gpu_mem_values),
        "avg_cpu_percent": average(cpu_values),
        "avg_gpu_util_percent": average(gpu_util_values),
        "total_detections": len(prediction_rows),
        "detections_by_class": dict(class_counter),
    }

    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)

    print("\nListo.")
    print(f"Resultados guardados en: {out_dir}")


if __name__ == "__main__":
    main()
