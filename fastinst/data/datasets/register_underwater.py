"""Registration helpers for COCO-format underwater instance datasets."""

import json
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances


_UIIS_SPLITS = {
    "uiis_train": ("train", "annotations/train.json"),
    "uiis_val": ("val", "annotations/val.json"),
}

_USIS10K_SPLITS = {
    "usis10k_train": (
        "train",
        "multi_class_annotations/multi_class_train_annotations.json",
    ),
    "usis10k_val": (
        "val",
        "multi_class_annotations/multi_class_val_annotations.json",
    ),
    "usis10k_test": (
        "test",
        "multi_class_annotations/multi_class_test_annotations.json",
    ),
}


def _read_coco_info(json_file):
    with open(json_file, "r", encoding="utf-8") as handle:
        coco = json.load(handle)

    images = coco.get("images")
    categories = coco.get("categories")
    if not isinstance(images, list):
        raise ValueError("COCO JSON is missing an 'images' list: {}".format(json_file))
    if not isinstance(categories, list) or not categories:
        raise ValueError(
            "COCO JSON must contain a non-empty 'categories' list: {}".format(json_file)
        )

    try:
        category_items = sorted(
            ((int(category["id"]), str(category["name"])) for category in categories),
            key=lambda item: item[0],
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Invalid COCO categories in {}: {}".format(json_file, error))

    category_ids = [category_id for category_id, _ in category_items]
    category_names = [name for _, name in category_items]
    if len(category_ids) != len(set(category_ids)):
        raise ValueError("Duplicate category ids in {}".format(json_file))
    if len(category_names) != len(set(category_names)):
        raise ValueError("Duplicate category names in {}".format(json_file))

    return {
        "num_images": len(images),
        "categories": category_items,
    }


def _register_dataset_family(family, root, splits):
    if not root:
        raise ValueError("A root directory is required for {}".format(family))

    root = os.path.abspath(os.path.expanduser(root))
    split_info = {}

    # Validate every path and JSON before modifying Detectron2's catalogs.
    for dataset_name, (image_subdir, json_subpath) in splits.items():
        image_root = os.path.join(root, image_subdir)
        json_file = os.path.join(root, json_subpath)
        if not os.path.isfile(json_file):
            raise FileNotFoundError("COCO annotation JSON not found: {}".format(json_file))
        if not os.path.isdir(image_root):
            raise FileNotFoundError("Image directory not found: {}".format(image_root))

        info = _read_coco_info(json_file)
        info.update(json_file=json_file, image_root=image_root)
        split_info[dataset_name] = info

    train_name = "{}_train".format(family)
    train_categories = split_info[train_name]["categories"]
    for dataset_name, info in split_info.items():
        if info["categories"] != train_categories:
            raise ValueError(
                "Category ids/names in {} do not match {}".format(dataset_name, train_name)
            )

    for dataset_name, info in split_info.items():
        if dataset_name in DatasetCatalog.list():
            metadata = MetadataCatalog.get(dataset_name)
            registered_json = os.path.abspath(metadata.get("json_file", ""))
            registered_images = os.path.abspath(metadata.get("image_root", ""))
            if (registered_json, registered_images) != (
                info["json_file"],
                info["image_root"],
            ):
                raise ValueError(
                    "Dataset '{}' is already registered with different paths".format(
                        dataset_name
                    )
                )
            continue

        register_coco_instances(
            dataset_name,
            {},
            info["json_file"],
            info["image_root"],
        )

    return {
        "family": family,
        "root": root,
        "categories": [name for _, name in train_categories],
        "splits": split_info,
    }


def register_uiis(root):
    """Register ``uiis_train`` and ``uiis_val`` from ``root``."""
    return _register_dataset_family("uiis", root, _UIIS_SPLITS)


def register_usis10k(root):
    """Register all USIS10K multi-class train/val/test splits from ``root``."""
    return _register_dataset_family("usis10k", root, _USIS10K_SPLITS)
