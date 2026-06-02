import argparse
import csv
from pathlib import Path

import yaml


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Check GT coverage for benchmark images.")
    parser.add_argument(
        "--config",
        default="configs/rfdetr_config.yaml",
        help="Path to the YAML configuration file.",
    )
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def resolve_path(path_value, project_root):
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return project_root / path


def main():
    args = parse_args()
    config = load_config(args.config)
    project_root = Path(config.get("project_root", ".")).expanduser().resolve()
    dataset_dir = resolve_path(config["output_dataset"], project_root)
    image_dir = dataset_dir / "images"
    gt_csv = dataset_dir / "gt_annotations.csv"

    images = sorted([p.name for p in image_dir.iterdir() if p.suffix.lower() in IMG_EXTS])
    gt_images = set()
    gt_rows = 0

    with open(gt_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gt_rows += 1
            gt_images.add(row["image_name"])

    missing = [image_name for image_name in images if image_name not in gt_images]

    print(f"Imagenes totales: {len(images)}")
    print(f"Imagenes con GT: {len(gt_images)}")
    print(f"GT boxes: {gt_rows}")
    print(f"Imagenes sin GT: {len(missing)}")

    if missing:
        print("Primeras imagenes sin GT:")
        for image_name in missing[:20]:
            print(image_name)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
