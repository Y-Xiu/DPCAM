import os
import csv
import random
import argparse
from collections import defaultdict

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Process VISA pcb1-3: center-crop to 1024, sample normals, keep anomalies, and split 7:1.5:1.5"
    )
    parser.add_argument(
        "--visa-root",
        type=str,
        default="./visa",
        help="Root directory containing pcb1/pcb2/pcb3",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="./visa_pcb123_1024_split",
        help="Output dataset root",
    )
    parser.add_argument(
        "--crop-size",
        type=int,
        default=1024,
        help="Center crop size",
    )
    parser.add_argument(
        "--normal-per-pcb",
        type=int,
        default=100,
        help="Number of normal images to sample from each pcb dataset",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    return parser.parse_args()


def ensure_dirs(output_root):
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(output_root, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(output_root, split, "masks"), exist_ok=True)


def center_crop_with_pad(arr, crop_size, fill_value=0):
    h, w = arr.shape[:2]
    ch = crop_size
    cw = crop_size

    if h >= ch:
        y1 = (h - ch) // 2
        y2 = y1 + ch
    else:
        y1, y2 = 0, h

    if w >= cw:
        x1 = (w - cw) // 2
        x2 = x1 + cw
    else:
        x1, x2 = 0, w

    cropped = arr[y1:y2, x1:x2]
    out_shape = (ch, cw) + (() if arr.ndim == 2 else (arr.shape[2],))
    out = np.full(out_shape, fill_value, dtype=arr.dtype)

    oy = (ch - cropped.shape[0]) // 2
    ox = (cw - cropped.shape[1]) // 2
    out[oy:oy + cropped.shape[0], ox:ox + cropped.shape[1]] = cropped
    return out


def split_label_tokens(label_text):
    tokens = [x.strip() for x in label_text.split(",") if x.strip()]
    return set(tokens)


def ratio_to_counts(n, ratios=(0.7, 0.15, 0.15)):
    a = int(n * ratios[0])
    b = int(n * ratios[1])
    c = n - a - b
    return {"train": a, "val": b, "test": c}


def stratified_multilabel_split(records, seed=42):
    """
    Approximate multilabel stratification for anomaly records.
    """
    rng = random.Random(seed)
    labels = sorted({l for r in records for l in r["labels"]})
    total_n = len(records)
    desired_total = ratio_to_counts(total_n)

    label_total = defaultdict(int)
    for r in records:
        for l in r["labels"]:
            label_total[l] += 1

    label_target = {
        s: {l: label_total[l] * (0.7 if s == "train" else 0.15) for l in labels}
        for s in ["train", "val", "test"]
    }
    label_current = {s: defaultdict(int) for s in ["train", "val", "test"]}
    split_current_n = {"train": 0, "val": 0, "test": 0}
    split_records = {"train": [], "val": [], "test": []}

    # Hard samples first: more labels, rarer labels
    def rarity_score(rec):
        return sum(1.0 / max(label_total[l], 1) for l in rec["labels"])

    order = list(records)
    order.sort(key=lambda r: (len(r["labels"]), rarity_score(r)), reverse=True)

    for rec in order:
        best_split = None
        best_score = None
        for s in ["train", "val", "test"]:
            label_need = 0.0
            for l in rec["labels"]:
                need = label_target[s][l] - label_current[s][l]
                if need > 0:
                    label_need += need

            total_need = desired_total[s] - split_current_n[s]
            score = (label_need, total_need)
            if best_score is None or score > best_score:
                best_score = score
                best_split = s

        # fallback tie random among equally scored splits
        candidates = []
        for s in ["train", "val", "test"]:
            label_need = 0.0
            for l in rec["labels"]:
                need = label_target[s][l] - label_current[s][l]
                if need > 0:
                    label_need += need
            total_need = desired_total[s] - split_current_n[s]
            if (label_need, total_need) == best_score:
                candidates.append(s)
        if len(candidates) > 1:
            best_split = rng.choice(candidates)

        split_records[best_split].append(rec)
        split_current_n[best_split] += 1
        for l in rec["labels"]:
            label_current[best_split][l] += 1

    return split_records


def save_item(item, split, output_root, crop_size):
    image = cv2.imread(item["image_path"], cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to read image: {item['image_path']}")

    if item["mask_path"]:
        mask = cv2.imread(item["mask_path"], cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Failed to read mask: {item['mask_path']}")
    else:
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

    image_crop = center_crop_with_pad(image, crop_size, fill_value=0)
    mask_crop = center_crop_with_pad(mask, crop_size, fill_value=0)

    name = item["save_name"]
    img_out = os.path.join(output_root, split, "images", f"{name}.jpg")
    mask_out = os.path.join(output_root, split, "masks", f"{name}.png")
    cv2.imwrite(img_out, image_crop)
    cv2.imwrite(mask_out, mask_crop)


def load_records_for_pcb(visa_root, pcb_name):
    csv_path = os.path.join(visa_root, pcb_name, "image_anno.csv")
    records = []
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_rel = row["image"].strip()
            label_text = row["label"].strip()
            mask_rel = row["mask"].strip()
            record = {
                "pcb": pcb_name,
                "image_rel": image_rel,
                "image_path": os.path.join(visa_root, image_rel),
                "mask_path": os.path.join(visa_root, mask_rel) if mask_rel else "",
                "label_text": label_text,
                "labels": split_label_tokens(label_text),
                "is_normal": label_text == "normal",
            }
            records.append(record)
    return records


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    ensure_dirs(args.output_root)

    all_normal = []
    all_anomaly = []
    metadata_rows = []

    for pcb in ["pcb1", "pcb2", "pcb3"]:
        records = load_records_for_pcb(args.visa_root, pcb)
        normals = [r for r in records if r["is_normal"]]
        anomalies = [r for r in records if not r["is_normal"]]

        if len(normals) > args.normal_per_pcb:
            normals = random.sample(normals, args.normal_per_pcb)

        all_normal.extend(normals)
        all_anomaly.extend(anomalies)

    # Split anomaly by multilabel stratification
    anomaly_splits = stratified_multilabel_split(all_anomaly, seed=args.seed)

    # Split normal by ratio
    normal_shuffled = list(all_normal)
    random.shuffle(normal_shuffled)
    normal_counts = ratio_to_counts(len(normal_shuffled))
    n_train = normal_counts["train"]
    n_val = normal_counts["val"]
    normal_splits = {
        "train": normal_shuffled[:n_train],
        "val": normal_shuffled[n_train:n_train + n_val],
        "test": normal_shuffled[n_train + n_val:],
    }

    # Merge and save
    split_all = {
        "train": anomaly_splits["train"] + normal_splits["train"],
        "val": anomaly_splits["val"] + normal_splits["val"],
        "test": anomaly_splits["test"] + normal_splits["test"],
    }

    for split in ["train", "val", "test"]:
        random.shuffle(split_all[split])

    counter = {"train": 0, "val": 0, "test": 0}
    for split in ["train", "val", "test"]:
        for item in split_all[split]:
            idx = counter[split]
            base_name = os.path.splitext(os.path.basename(item["image_rel"]))[0]
            save_name = f"{split}_{item['pcb']}_{idx:05d}_{base_name}"
            item["save_name"] = save_name
            save_item(item, split, args.output_root, args.crop_size)
            metadata_rows.append({
                "split": split,
                "save_name": save_name,
                "pcb": item["pcb"],
                "label": item["label_text"],
                "image_src": item["image_rel"],
                "mask_src": item["mask_path"].replace(args.visa_root + os.sep, "") if item["mask_path"] else "",
            })
            counter[split] += 1

    # Save metadata
    meta_csv = os.path.join(args.output_root, "metadata.csv")
    with open(meta_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["split", "save_name", "pcb", "label", "image_src", "mask_src"]
        )
        writer.writeheader()
        writer.writerows(metadata_rows)

    # Save summary
    summary_txt = os.path.join(args.output_root, "summary.txt")
    split_label_count = {s: defaultdict(int) for s in ["train", "val", "test"]}
    for r in metadata_rows:
        split_label_count[r["split"]][r["label"]] += 1

    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write("VISA pcb1-3 processed dataset summary\n")
        f.write(f"crop_size={args.crop_size}, normal_per_pcb={args.normal_per_pcb}, seed={args.seed}\n\n")
        for s in ["train", "val", "test"]:
            f.write(f"[{s}] total={counter[s]}\n")
            for label, c in sorted(split_label_count[s].items(), key=lambda x: x[0]):
                f.write(f"  {label}: {c}\n")
            f.write("\n")

    print("Done.")
    print(f"Output: {args.output_root}")
    print(f"Metadata: {meta_csv}")
    print(f"Summary: {summary_txt}")


if __name__ == "__main__":
    main()
