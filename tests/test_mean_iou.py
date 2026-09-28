# Copyright 2026 The HuggingFace Evaluate Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

import importlib

import numpy as np
import pytest
from PIL import Image

import evaluate


@pytest.fixture
def mean_iou_metric():
    return evaluate.load("./metrics/mean_iou")


@pytest.mark.parametrize("mapping", [{0: 1, 1: 0}, {0: 1, 1: 2, 2: 0}])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("as_image", [False, True])
def test_compute_label_map_is_simultaneous(mean_iou_metric, mapping, reverse, as_image):
    reference = np.arange(len(mapping), dtype=np.uint8).reshape(1, -1)
    prediction = np.array([[mapping[int(value)] for value in reference[0]]], dtype=np.uint8)
    original = reference.copy()
    if reverse:
        mapping = dict(reversed(list(mapping.items())))
    result = mean_iou_metric.compute(
        predictions=[Image.fromarray(prediction) if as_image else prediction],
        references=[Image.fromarray(reference) if as_image else reference],
        num_labels=len(mapping),
        ignore_index=255,
        label_map=mapping,
    )
    assert result["mean_iou"] == 1.0
    assert result["overall_accuracy"] == 1.0
    np.testing.assert_array_equal(result["per_category_iou"], np.ones(len(mapping)))
    np.testing.assert_array_equal(reference, original)


@pytest.mark.parametrize("mapping", [{0: 1, 1: 0}, {0: 1, 1: 2, 2: 0}])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("read_only", [False, True])
def test_intersection_does_not_modify_input(mean_iou_metric, mapping, reverse, read_only):
    module = importlib.import_module(mean_iou_metric.__class__.__module__)
    reference = np.arange(len(mapping), dtype=np.uint8).reshape(1, -1)
    prediction = np.array([[mapping[int(value)] for value in reference[0]]], dtype=np.uint8)
    original = reference.copy()
    if reverse:
        mapping = dict(reversed(list(mapping.items())))
    if read_only:
        reference.setflags(write=False)
    intersection, union, _, _ = module.intersect_and_union(
        prediction, reference, num_labels=len(mapping), ignore_index=255, label_map=mapping
    )
    np.testing.assert_array_equal(reference, original)
    np.testing.assert_array_equal(intersection, np.ones(len(mapping)))
    np.testing.assert_array_equal(union, np.ones(len(mapping)))


@pytest.mark.parametrize("mapping", [None, {}, {1: 2, 2: 1}])
@pytest.mark.parametrize("as_image", [False, True])
def test_label_map_preserves_reduction_and_ignored_pixels(mean_iou_metric, mapping, as_image):
    reference = np.array([[0, 1, 2, 255]], dtype=np.uint8)
    prediction = np.array([[7, 1, 0, 7]] if mapping else [[7, 0, 1, 7]], dtype=np.uint8)
    result = mean_iou_metric.compute(
        predictions=[Image.fromarray(prediction) if as_image else prediction],
        references=[Image.fromarray(reference) if as_image else reference],
        num_labels=2,
        ignore_index=255,
        reduce_labels=True,
        label_map=mapping,
    )
    assert result["mean_iou"] == 1.0
    np.testing.assert_array_equal(result["per_category_iou"], [1.0, 1.0])
    np.testing.assert_array_equal(reference, [[0, 1, 2, 255]])
