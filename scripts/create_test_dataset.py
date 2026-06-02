import argparse
import csv
import json
import random
import shutil
from pathlib import Path

import yaml
from PIL import Image


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a balanced benchmark dataset from multiple detection datasets."
    )
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


def safe_name(name):
    return (
        name.replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("[", "")
        .replace("]", "")
    )


def is_rgb_readable(path):
    try:
        with Image.open(path) as im:
            im.convert("RGB")
        return True
    except Exception:
        return False


def copy_json(json_path, dataset_id, output_annotations_dir):
    if not json_path.exists():
        return None

    dst = output_annotations_dir / f"{dataset_id}_{json_path.parent.name}_{json_path.name}"
    shutil.copy2(json_path, dst)
    return dst


def load_coco_candidates(dataset_id, dataset_dir, json_paths, output_annotations_dir):
    candidates = []

    for json_path in json_paths:
        if not json_path.exists():
            print(f"[WARN] No existe: {json_path}")
            continue

        with open(json_path, "r") as f:
            coco = json.load(f)

        images = coco.get("images", [])
        annotations = coco.get("annotations", [])
        image_ids_with_ann = {ann["image_id"] for ann in annotations}

        for img in images:
            img_id = img["id"]
            rel_path = img["file_name"]

            if img_id not in image_ids_with_ann:
                continue

            abs_path = dataset_dir / rel_path
            if not abs_path.exists():
                abs_path = json_path.parent / rel_path

            if not abs_path.exists():
                continue

            if abs_path.suffix.lower() not in IMG_EXTS:
                continue

            if not is_rgb_readable(abs_path):
                continue

            candidates.append(
                {
                    "dataset_id": dataset_id,
                    "abs_path": abs_path,
                    "original_relative_path": rel_path,
                    "original_file_name": abs_path.name,
                    "annotation_format": "coco",
                    "annotation_path": str(json_path),
                }
            )

        copy_json(json_path, dataset_id, output_annotations_dir)

    return candidates


def find_yolo_names_file(dataset_dir):
    names_files = list(dataset_dir.rglob("*.names"))
    return names_files[0] if names_files else None


def load_yolo_class_names(names_path):
    if names_path and names_path.exists():
        return [x.strip() for x in names_path.read_text().splitlines() if x.strip()]
    return []


def find_yolo_label_for_image(img_path, dataset_dir):
    label_name = img_path.with_suffix(".txt").name
    direct = img_path.with_suffix(".txt")

    if direct.exists():
        return direct

    labels = list(dataset_dir.rglob(label_name))
    return labels[0] if labels else None


def load_yolo_candidates(dataset_id, dataset_dir, output_annotations_dir):
    candidates = []

    for img_path in dataset_dir.rglob("*"):
        if img_path.suffix.lower() not in IMG_EXTS:
            continue

        lower_parts = [p.lower() for p in img_path.parts]
        if any("mask" in p or "label" in p or "annotation" in p for p in lower_parts):
            continue

        if not is_rgb_readable(img_path):
            continue

        label_path = find_yolo_label_for_image(img_path, dataset_dir)
        if not label_path or not label_path.exists():
            continue

        candidates.append(
            {
                "dataset_id": dataset_id,
                "abs_path": img_path,
                "original_relative_path": str(img_path.relative_to(dataset_dir)),
                "original_file_name": img_path.name,
                "annotation_format": "yolo_darknet",
                "annotation_path": str(label_path),
            }
        )

    names_file = find_yolo_names_file(dataset_dir)
    if names_file:
        shutil.copy2(names_file, output_annotations_dir / f"{dataset_id}_{names_file.name}")

    return candidates


def select_and_copy(dataset_id, candidates, output_images_dir, n):
    if len(candidates) < n:
        raise ValueError(
            f"{dataset_id} tiene solo {len(candidates)} imagenes con GT, se necesitan {n}"
        )

    selected = random.sample(candidates, n)
    map_rows = []

    for i, item in enumerate(selected, start=1):
        original_name = safe_name(item["original_file_name"])
        new_name = f"{dataset_id}_{i:03d}_{original_name}"
        dst = output_images_dir / new_name

        shutil.copy2(item["abs_path"], dst)

        map_rows.append(
            {
                "new_image_name": new_name,
                "dataset_id": dataset_id,
                "original_abs_path": str(item["abs_path"]),
                "original_relative_path": item["original_relative_path"],
                "original_file_name": item["original_file_name"],
                "annotation_format": item["annotation_format"],
                "annotation_path": item["annotation_path"],
            }
        )

    return map_rows


def build_gt_from_map(map_rows, output_images_dir, dataset_dirs):
    gt_rows = []
    coco_cache = {}
    yolo_names_cache = {}

    for row in map_rows:
        image_name = row["new_image_name"]
        dataset_id = row["dataset_id"]

        if row["annotation_format"] == "coco":
            json_path = Path(row["annotation_path"])

            if json_path not in coco_cache:
                with open(json_path, "r") as f:
                    coco = json.load(f)

                cats = {c["id"]: c["name"] for c in coco["categories"]}
                imgs = {img["file_name"]: img for img in coco["images"]}
                anns_by_img_id = {}

                for ann in coco["annotations"]:
                    anns_by_img_id.setdefault(ann["image_id"], []).append(ann)

                coco_cache[json_path] = {
                    "cats": cats,
                    "imgs": imgs,
                    "anns_by_img_id": anns_by_img_id,
                }

            data = coco_cache[json_path]
            img_info = data["imgs"].get(row["original_relative_path"])

            if img_info is None:
                matches = [
                    img
                    for fn, img in data["imgs"].items()
                    if Path(fn).name == row["original_file_name"]
                ]
                if len(matches) == 1:
                    img_info = matches[0]

            if img_info is None:
                continue

            for ann in data["anns_by_img_id"].get(img_info["id"], []):
                x, y, w, h = ann["bbox"]
                class_name = data["cats"].get(ann["category_id"], str(ann["category_id"]))

                gt_rows.append(
                    {
                        "image_name": image_name,
                        "dataset_id": dataset_id,
                        "class_name": class_name,
                        "x1": x,
                        "y1": y,
                        "x2": x + w,
                        "y2": y + h,
                        "bbox_width": w,
                        "bbox_height": h,
                        "bbox_area": w * h,
                        "source_format": "coco",
                    }
                )

        elif row["annotation_format"] == "yolo_darknet":
            img_path = output_images_dir / image_name
            label_path = Path(row["annotation_path"])
            dataset_dir = dataset_dirs[dataset_id]

            with Image.open(img_path) as im:
                img_w, img_h = im.size

            if dataset_id not in yolo_names_cache:
                yolo_names_cache[dataset_id] = load_yolo_class_names(
                    find_yolo_names_file(dataset_dir)
                )
            class_names = yolo_names_cache[dataset_id]

            for line in label_path.read_text().splitlines():
                parts = line.strip().split()
                if len(parts) != 5:
                    continue

                cls_id = int(float(parts[0]))
                xc, yc, bw, bh = map(float, parts[1:])

                x1 = (xc - bw / 2) * img_w
                y1 = (yc - bh / 2) * img_h
                x2 = (xc + bw / 2) * img_w
                y2 = (yc + bh / 2) * img_h
                class_name = (
                    class_names[cls_id]
                    if cls_id < len(class_names)
                    else f"class_{cls_id}"
                )

                gt_rows.append(
                    {
                        "image_name": image_name,
                        "dataset_id": dataset_id,
                        "class_name": class_name,
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "bbox_width": x2 - x1,
                        "bbox_height": y2 - y1,
                        "bbox_area": (x2 - x1) * (y2 - y1),
                        "source_format": "yolo_darknet",
                    }
                )

    return gt_rows


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    config = load_config(args.config)
    project_root = project_root_from_config(config)

    output_dataset = resolve_path(config["output_dataset"], project_root)
    output_images_dir = output_dataset / "images"
    output_annotations_dir = output_dataset / "annotations_original"
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_annotations_dir.mkdir(parents=True, exist_ok=True)

    map_csv = output_dataset / "image_source_map.csv"
    gt_csv = output_dataset / "gt_annotations.csv"
    num_images = int(config.get("num_images_per_dataset", 100))
    random.seed(int(config.get("seed", 42)))

    dataset_dirs = {}
    all_map_rows = []

    for dataset_id, dataset_config in config["datasets"].items():
        dataset_dir = resolve_path(dataset_config["root"], project_root)
        dataset_dirs[dataset_id] = dataset_dir

        print(f"Cargando candidatos {dataset_id} ({dataset_config['name']})...")

        if dataset_config["format"] == "coco":
            json_paths = [
                dataset_dir / annotation
                for annotation in dataset_config.get("annotations", [])
            ]
            candidates = load_coco_candidates(
                dataset_id, dataset_dir, json_paths, output_annotations_dir
            )
        elif dataset_config["format"] == "yolo_darknet":
            candidates = load_yolo_candidates(
                dataset_id, dataset_dir, output_annotations_dir
            )
        else:
            raise ValueError(f"Formato no soportado: {dataset_config['format']}")

        print(f"{dataset_id} candidatos con GT: {len(candidates)}")
        all_map_rows.extend(
            select_and_copy(dataset_id, candidates, output_images_dir, num_images)
        )

    write_csv(
        map_csv,
        all_map_rows,
        [
            "new_image_name",
            "dataset_id",
            "original_abs_path",
            "original_relative_path",
            "original_file_name",
            "annotation_format",
            "annotation_path",
        ],
    )

    gt_rows = build_gt_from_map(all_map_rows, output_images_dir, dataset_dirs)
    write_csv(
        gt_csv,
        gt_rows,
        [
            "image_name",
            "dataset_id",
            "class_name",
            "x1",
            "y1",
            "x2",
            "y2",
            "bbox_width",
            "bbox_height",
            "bbox_area",
            "source_format",
        ],
    )

    images_with_gt = {r["image_name"] for r in gt_rows}
    missing = [
        r["new_image_name"]
        for r in all_map_rows
        if r["new_image_name"] not in images_with_gt
    ]

    print("\nListo.")
    print(f"Imagenes copiadas en: {output_images_dir}")
    print(f"Mapa guardado en: {map_csv}")
    print(f"GT guardado en: {gt_csv}")
    print(f"Total imagenes: {len(list(output_images_dir.iterdir()))}")
    print(f"Total GT boxes: {len(gt_rows)}")
    print(f"Imagenes con GT: {len(images_with_gt)}")
    print(f"Imagenes sin GT: {len(missing)}")

    if missing:
        print("Primeras sin GT:")
        for image_name in missing[:20]:
            print(image_name)


if __name__ == "__main__":
    main()
