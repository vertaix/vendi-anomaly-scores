#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class ImageFolderWithWNID(Dataset):
    def __init__(self, root: Path):
        self.root = Path(root)
        self.paths = sorted(self.root.glob("*/*.JPEG"))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        wnid = path.parent.name
        image_id = path.stem
        img = Image.open(path).convert("RGB")
        return img, str(path), wnid, image_id


def collate_fn(batch):
    imgs, paths, wnids, image_ids = zip(*batch)
    return list(imgs), list(paths), list(wnids), list(image_ids)


def build_wnid_mapping():
    try:
        from datasets import load_dataset
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "This helper script requires the optional 'datasets' package. "
            "Install extra ImageNet-prep dependencies before running it."
        ) from exc

    ds = load_dataset("visual-layer/imagenet-1k-vl-enriched", split="validation")
    wnid_to_label = {}
    for ex in tqdm(ds, desc="Building wnid -> label mapping"):
        wnid = ex["image_id"].split("_")[0]
        if "label" in ex and "label" in ds.features:
            label_name = ds.features["label"].int2str(ex["label"])
        elif "class_name" in ex:
            label_name = ex["class_name"]
        elif "category" in ex:
            label_name = ex["category"]
        else:
            label_name = "UNKNOWN"
        wnid_to_label[wnid] = label_name
    return wnid_to_label


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/imagenet/imagenet_1k"))
    parser.add_argument("--split", choices=["train", "val", "both"], default="both")
    parser.add_argument("--out-dir", type=Path, default=Path("data/imagenet/dinov3_vitb16_embeddings"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--model-name",
        type=str,
        default="facebook/dinov3-vitb16-pretrain-lvd1689m",
    )
    parser.add_argument(
        "--skip-label-map",
        action="store_true",
        help="Skip downloading the optional WNID-to-label mapping helper dataset.",
    )
    args = parser.parse_args()

    try:
        from transformers import AutoImageProcessor, AutoModel
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "This helper script requires optional packages such as 'transformers'. "
            "See the README ImageNet section for the extra dependencies."
        ) from exc

    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading DINOv3 processor/model...")
    processor = AutoImageProcessor.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name).to(device).eval()

    if args.skip_label_map:
        wnid_to_label = {}
    else:
        print("Building class-label mapping...")
        wnid_to_label = build_wnid_mapping()

    with (args.out_dir / "wnid_to_label.json").open("w") as handle:
        json.dump(wnid_to_label, handle, indent=2)

    with (args.out_dir / "wnid_to_label.txt").open("w") as handle:
        for wnid in sorted(wnid_to_label):
            handle.write(f"{wnid}\t{wnid_to_label[wnid]}\n")

    splits = ["train", "val"] if args.split == "both" else [args.split]

    total_images = 0
    for split in splits:
        total_images += len(list((args.data_root / split).glob("*/*.JPEG")))
    print(f"Total images: {total_images}")

    dummy_img = Image.new("RGB", (224, 224))
    dummy_inputs = processor(images=[dummy_img], return_tensors="pt")
    dummy_inputs = {k: v.to(device) for k, v in dummy_inputs.items()}
    dummy_outputs = model(**dummy_inputs)
    embed_dim = int(dummy_outputs.last_hidden_state[:, 0].shape[1])
    print("Embedding dim:", embed_dim)

    safe_model_name = args.model_name.replace("/", "_")
    embeddings_path = args.out_dir / f"{safe_model_name}_embeddings.npy"
    metadata_path = args.out_dir / f"{safe_model_name}_metadata.csv"

    embeddings = np.lib.format.open_memmap(
        embeddings_path,
        mode="w+",
        dtype=np.float32,
        shape=(total_images, embed_dim),
    )

    offset = 0
    with metadata_path.open("w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["index", "split", "path", "image_id", "wnid", "label_name"])

        for split in splits:
            dataset = ImageFolderWithWNID(args.data_root / split)
            loader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
                collate_fn=collate_fn,
            )

            for imgs, paths, wnids, image_ids in tqdm(loader, desc=f"Embedding {split}"):
                inputs = processor(images=imgs, return_tensors="pt")
                inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}
                outputs = model(**inputs)

                feats = outputs.last_hidden_state[:, 0]
                feats = torch.nn.functional.normalize(feats, dim=1)
                feats = feats.float().cpu().numpy()
                bsz = feats.shape[0]

                embeddings[offset:offset + bsz] = feats
                for j in range(bsz):
                    idx = offset + j
                    writer.writerow(
                        [
                            idx,
                            split,
                            paths[j],
                            image_ids[j],
                            wnids[j],
                            wnid_to_label.get(wnids[j], "UNKNOWN"),
                        ]
                    )
                offset += bsz

    print("Done.")
    print("Embeddings:", embeddings_path)
    print("Metadata:", metadata_path)
    print("Class mapping:", args.out_dir / "wnid_to_label.txt")


if __name__ == "__main__":
    main()
