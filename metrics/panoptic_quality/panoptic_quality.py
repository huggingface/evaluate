# Copyright 2022 The HuggingFace Evaluate Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Panoptic Quality (PQ) metric.

Entirely based on https://github.com/cocodataset/panopticapi/blob/master/panopticapi/evaluation.py.
"""

from collections import defaultdict
import functools
import json
import multiprocessing
import io
import os
import traceback

import numpy as np
from PIL import Image

import datasets
import evaluate


_DESCRIPTION = """...
"""

_KWARGS_DESCRIPTION = """
Args:
    predictions (`List[ndarray]`):
        List of predicted segmentation maps, each of shape (height, width). Each segmentation map can be of a different size.
    references (`List[ndarray]`):
        List of ground truth segmentation maps, each of shape (height, width). Each segmentation map can be of a different size.
    predicted_annotations (`List[ndarray]`):
        List of predicted annotations (segments info).
    reference_annotations (`List[ndarray]`):
        List of reference annotations (segments info).
    output_dir (`str`):
        Path to the output directory.
    categories (`dict`):
        Dictionary mapping category IDs to something like {'name': 'wall', 'id': 0, 'isthing': 0, 'color': [120, 120, 120]}.
        Example here: https://github.com/cocodataset/panopticapi/blob/master/panoptic_coco_categories.json.

Returns:
    `Dict[str, float | ndarray]` comprising various elements:
    - *panoptic_quality* (`float`):
       Panoptic quality score.
    ...

Examples:

    >>> import numpy as np

    >>> panoptic_quality = evaluate.load("panoptic_quality")

    >>> # TODO
"""

_CITATION = """..."""


# The decorator is used to prints an error trhown inside process
def get_traceback(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            print("Caught exception in worker thread:")
            traceback.print_exc()
            raise e

    return wrapper


def rgb2id(color):
    if isinstance(color, np.ndarray) and len(color.shape) == 3:
        if color.dtype == np.uint8:
            color = color.astype(np.int32)
        return color[:, :, 0] + 256 * color[:, :, 1] + 256 * 256 * color[:, :, 2]
    return int(color[0] + 256 * color[1] + 256 * 256 * color[2])


def id2rgb(id_map):
    if isinstance(id_map, np.ndarray):
        id_map_copy = id_map.copy()
        rgb_shape = tuple(list(id_map.shape) + [3])
        rgb_map = np.zeros(rgb_shape, dtype=np.uint8)
        for i in range(3):
            rgb_map[..., i] = id_map_copy % 256
            id_map_copy //= 256
        return rgb_map
    color = []
    for _ in range(3):
        color.append(id_map % 256)
        id_map //= 256
    return color


OFFSET = 256 * 256 * 256
VOID = 0


class PQStatCat:
    def __init__(self):
        self.iou = 0.0
        self.tp = 0
        self.fp = 0
        self.fn = 0

    def __iadd__(self, pq_stat_cat):
        self.iou += pq_stat_cat.iou
        self.tp += pq_stat_cat.tp
        self.fp += pq_stat_cat.fp
        self.fn += pq_stat_cat.fn
        return self


class PQStat:
    def __init__(self):
        self.pq_per_cat = defaultdict(PQStatCat)

    def __getitem__(self, i):
        return self.pq_per_cat[i]

    def __iadd__(self, pq_stat):
        for label, pq_stat_cat in pq_stat.pq_per_cat.items():
            self.pq_per_cat[label] += pq_stat_cat
        return self

    def pq_average(self, categories, isthing):
        pq, sq, rq, n = 0, 0, 0, 0
        per_class_results = {}
        for label, label_info in categories.items():
            if isthing is not None:
                cat_isthing = label_info["isthing"] == 1
                if isthing != cat_isthing:
                    continue
            iou = self.pq_per_cat[label].iou
            tp = self.pq_per_cat[label].tp
            fp = self.pq_per_cat[label].fp
            fn = self.pq_per_cat[label].fn
            if tp + fp + fn == 0:
                per_class_results[label] = {"pq": 0.0, "sq": 0.0, "rq": 0.0}
                continue
            n += 1
            pq_class = iou / (tp + 0.5 * fp + 0.5 * fn)
            sq_class = iou / tp if tp != 0 else 0
            rq_class = tp / (tp + 0.5 * fp + 0.5 * fn)
            per_class_results[label] = {"pq": pq_class, "sq": sq_class, "rq": rq_class}
            pq += pq_class
            sq += sq_class
            rq += rq_class

        return {"pq": pq / n, "sq": sq / n, "rq": rq / n, "n": n}, per_class_results


@get_traceback
def pq_compute_single_core(proc_id, annotation_set, gt_folder, pred_folder, categories):
    pq_stat = PQStat()

    idx = 0
    for gt_ann, pred_ann in annotation_set:
        if idx % 100 == 0:
            print("Core: {}, {} from {} images processed".format(proc_id, idx, len(annotation_set)))
        idx += 1

        pan_gt = np.array(Image.open(os.path.join(gt_folder, gt_ann["file_name"])), dtype=np.uint32)
        pan_gt = rgb2id(pan_gt)
        pan_pred = np.array(Image.open(os.path.join(pred_folder, pred_ann["file_name"])), dtype=np.uint32)
        pan_pred = rgb2id(pan_pred)

        gt_segms = {el["id"]: el for el in gt_ann["segments_info"]}
        pred_segms = {el["id"]: el for el in pred_ann["segments_info"]}

        # predicted segments area calculation + prediction sanity checks
        pred_labels_set = set(el["id"] for el in pred_ann["segments_info"])
        labels, labels_cnt = np.unique(pan_pred, return_counts=True)
        for label, label_cnt in zip(labels, labels_cnt):
            if label not in pred_segms:
                if label == VOID:
                    continue
                raise KeyError(
                    "In the image with ID {} segment with ID {} is presented in PNG and not presented in JSON.".format(
                        gt_ann["image_id"], label
                    )
                )
            pred_segms[label]["area"] = label_cnt
            pred_labels_set.remove(label)
            if pred_segms[label]["category_id"] not in categories:
                raise KeyError(
                    "In the image with ID {} segment with ID {} has unknown category_id {}.".format(
                        gt_ann["image_id"], label, pred_segms[label]["category_id"]
                    )
                )
        if len(pred_labels_set) != 0:
            raise KeyError(
                "In the image with ID {} the following segment IDs {} are presented in JSON and not presented in PNG.".format(
                    gt_ann["image_id"], list(pred_labels_set)
                )
            )

        # confusion matrix calculation
        pan_gt_pred = pan_gt.astype(np.uint64) * OFFSET + pan_pred.astype(np.uint64)
        gt_pred_map = {}
        labels, labels_cnt = np.unique(pan_gt_pred, return_counts=True)
        for label, intersection in zip(labels, labels_cnt):
            gt_id = label // OFFSET
            pred_id = label % OFFSET
            gt_pred_map[(gt_id, pred_id)] = intersection

        # count all matched pairs
        gt_matched = set()
        pred_matched = set()
        for label_tuple, intersection in gt_pred_map.items():
            gt_label, pred_label = label_tuple
            if gt_label not in gt_segms:
                continue
            if pred_label not in pred_segms:
                continue
            if gt_segms[gt_label]["iscrowd"] == 1:
                continue
            if gt_segms[gt_label]["category_id"] != pred_segms[pred_label]["category_id"]:
                continue

            union = (
                pred_segms[pred_label]["area"]
                + gt_segms[gt_label]["area"]
                - intersection
                - gt_pred_map.get((VOID, pred_label), 0)
            )
            iou = intersection / union
            if iou > 0.5:
                pq_stat[gt_segms[gt_label]["category_id"]].tp += 1
                pq_stat[gt_segms[gt_label]["category_id"]].iou += iou
                gt_matched.add(gt_label)
                pred_matched.add(pred_label)

        # count false positives
        crowd_labels_dict = {}
        for gt_label, gt_info in gt_segms.items():
            if gt_label in gt_matched:
                continue
            # crowd segments are ignored
            if gt_info["iscrowd"] == 1:
                crowd_labels_dict[gt_info["category_id"]] = gt_label
                continue
            pq_stat[gt_info["category_id"]].fn += 1

        # count false positives
        for pred_label, pred_info in pred_segms.items():
            if pred_label in pred_matched:
                continue
            # intersection of the segment with VOID
            intersection = gt_pred_map.get((VOID, pred_label), 0)
            # plus intersection with corresponding CROWD region if it exists
            if pred_info["category_id"] in crowd_labels_dict:
                intersection += gt_pred_map.get((crowd_labels_dict[pred_info["category_id"]], pred_label), 0)
            # predicted segment is ignored if more than half of the segment correspond to VOID and CROWD regions
            if intersection / pred_info["area"] > 0.5:
                continue
            pq_stat[pred_info["category_id"]].fp += 1
    print("Core: {}, all {} images processed".format(proc_id, len(annotation_set)))
    return pq_stat


def pq_compute_multi_core(matched_annotations_list, gt_folder, pred_folder, categories):
    cpu_num = multiprocessing.cpu_count()
    annotations_split = np.array_split(matched_annotations_list, cpu_num)
    print("Number of cores: {}, images per core: {}".format(cpu_num, len(annotations_split[0])))
    workers = multiprocessing.Pool(processes=cpu_num)
    processes = []
    for proc_id, annotation_set in enumerate(annotations_split):
        p = workers.apply_async(pq_compute_single_core, (proc_id, annotation_set, gt_folder, pred_folder, categories))
        processes.append(p)
    pq_stat = PQStat()
    for p in processes:
        pq_stat += p.get()
    return pq_stat


def pq_compute(gt_json_file, pred_json_file, gt_folder=None, pred_folder=None):
    with open(gt_json_file, "r") as f:
        gt_json = json.load(f)
    with open(pred_json_file, "r") as f:
        pred_json = json.load(f)

    if gt_folder is None:
        gt_folder = gt_json_file.replace(".json", "")
    if pred_folder is None:
        pred_folder = pred_json_file.replace(".json", "")
    categories = {el["id"]: el for el in gt_json["categories"]}

    print("Evaluation panoptic segmentation metrics:")
    print("Ground truth:")
    print("\tSegmentation folder: {}".format(gt_folder))
    print("\tJSON file: {}".format(gt_json_file))
    print("Prediction:")
    print("\tSegmentation folder: {}".format(pred_folder))
    print("\tJSON file: {}".format(pred_json_file))

    if not os.path.isdir(gt_folder):
        raise Exception("Folder {} with ground truth segmentations doesn't exist".format(gt_folder))
    if not os.path.isdir(pred_folder):
        raise Exception("Folder {} with predicted segmentations doesn't exist".format(pred_folder))

    pred_annotations = {el["image_id"]: el for el in pred_json["annotations"]}
    matched_annotations_list = []
    for gt_ann in gt_json["annotations"]:
        image_id = gt_ann["image_id"]
        if image_id not in pred_annotations:
            raise Exception("no prediction for the image with id: {}".format(image_id))
        matched_annotations_list.append((gt_ann, pred_annotations[image_id]))

    pq_stat = pq_compute_multi_core(matched_annotations_list, gt_folder, pred_folder, categories)

    metrics = [("All", None), ("Things", True), ("Stuff", False)]
    results = {}
    for name, isthing in metrics:
        results[name], per_class_results = pq_stat.pq_average(categories, isthing=isthing)
        if name == "All":
            results["per_class"] = per_class_results
    print("{:10s}| {:>5s}  {:>5s}  {:>5s} {:>5s}".format("", "PQ", "SQ", "RQ", "N"))
    print("-" * (10 + 7 * 4))

    for name, _isthing in metrics:
        print(
            "{:10s}| {:5.1f}  {:5.1f}  {:5.1f} {:5d}".format(
                name,
                100 * results[name]["pq"],
                100 * results[name]["sq"],
                100 * results[name]["rq"],
                results[name]["n"],
            )
        )


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class PanopticQuality(evaluate.Metric):
    def _info(self):
        return evaluate.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                # 1st Seq - height dim, 2nd - width dim
                {
                    "predictions": datasets.Sequence(datasets.Sequence(datasets.Value("uint16"))),
                }
            ),
            reference_urls=[
                "https://github.com/open-mmlab/mmsegmentation/blob/71c201b1813267d78764f306a297ca717827c4bf/mmseg/core/evaluation/metrics.py"
            ],
        )

    def _compute(
        self,
        predictions=None,
        references=None,
        predicted_annotations=None,
        image_ids=None,
        # references,
        # reference_annotations,
        output_dir=None,
        gt_folder=None,
        gt_json=None,
    ):
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        # step 1: dump predicted segmentations to folder
        for seg_img, image_id in zip(predictions, image_ids):
            seg_img = np.array(seg_img)
            print("Shape of seg_img:", seg_img.shape)
            seg_img = Image.fromarray(id2rgb(seg_img))

            with io.BytesIO() as out:
                seg_img.save(out, format="PNG")
            
            file_name = f"{image_id:012d}.png"
            with open(os.path.join(output_dir, file_name), "wb") as f:
                f.write(out.getvalue())

        # step 2: create predictions JSON file
        # predicted_annotations is a list of segments_info
        json_data = {"annotations": predicted_annotations}
        predictions_json = os.path.join(output_dir, "predictions.json")
        with open(predictions_json, "w") as f:
            f.write(json.dumps(json_data))
                
        # step 5: compute PQ
        result = pq_compute(gt_json, predictions_json, gt_folder=gt_folder, pred_folder=output_dir)
        
        return result


# # step 1: create ground truth JSON file
# gt_json_data = {"annotations": reference_annotations, "categories": categories}
# gt_json = os.path.join(output_dir, "gt.json")
# with open(gt_json, "w") as f:
#     f.write(json.dumps(gt_json_data))

# # step 4
        # gt_folder = os.path.join(output_dir, "gt")
        # if not os.path.exists(gt_folder):
        #     os.mkdir(gt_folder)
        # for ref in reference_annotations:
        #     image_id = ref["image_id"]
        #     file_name = f"{image_id:012d}.png"
        #     with open(os.path.join(gt_folder, file_name), "wb") as f:
        #         f.write(ref["segmentation"])