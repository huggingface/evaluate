from dataclasses import asdict

import pytest
from datasets import Features, Sequence, Value

from evaluate import MetricInfo


@pytest.mark.parametrize(
    "features,format",
    [
        (Features({"predictions": Value("int64")}), None),
        (Features({"predictions": Sequence(Value("string"))}), None),
        ([Features({"predictions": Value("int64")}), Features({"predictions": Value("string")})], None),
        (Features({"predictions": Value("int64")}), "numpy"),
    ],
)
def test_metric_info_feature_roundtrip(tmp_path, features, format):
    info = MetricInfo(description="Local metric", citation="", features=features, format=format)
    serialized = asdict(info)
    restored = MetricInfo.from_dict(serialized)
    assert restored == info
    assert serialized == asdict(info)

    info.write_to_directory(tmp_path)
    restored = MetricInfo.from_directory(tmp_path)
    assert restored == info
    if isinstance(features, list):
        assert all(isinstance(feature, Features) for feature in restored.features)
    else:
        assert isinstance(restored.features, Features)
        assert restored.features.arrow_schema == features.arrow_schema


@pytest.mark.parametrize(
    "features",
    [
        Features({"predictions": Value("int64")}),
        [Features({"predictions": Value("int64")}), Features({"predictions": Value("string")})],
        None,
    ],
)
def test_metric_info_from_dict_accepts_feature_objects(features):
    info_dict = {"description": "Local metric", "citation": "", "features": features}
    restored = MetricInfo.from_dict(info_dict)
    assert restored.features == features
    assert info_dict["features"] is features
