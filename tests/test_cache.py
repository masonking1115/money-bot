import pandas as pd

from moneybot.cache import Cache


def test_json_set_get_roundtrip(tmp_path):
    cache = Cache(tmp_path)
    cache.set_json("k1", {"a": 1, "b": [2, 3]})
    assert cache.get_json("k1") == {"a": 1, "b": [2, 3]}


def test_json_get_missing_returns_none(tmp_path):
    cache = Cache(tmp_path)
    assert cache.get_json("nope") is None


def test_dataframe_set_get_roundtrip(tmp_path):
    cache = Cache(tmp_path)
    df = pd.DataFrame({"close": [1.0, 2.0], "volume": [10, 20]})
    cache.set_dataframe("bars:NVDA", df)
    out = cache.get_dataframe("bars:NVDA")
    pd.testing.assert_frame_equal(out, df)


def test_dataframe_get_missing_returns_none(tmp_path):
    cache = Cache(tmp_path)
    assert cache.get_dataframe("missing") is None


def test_persists_across_instances(tmp_path):
    Cache(tmp_path).set_json("persist", {"x": 1})
    assert Cache(tmp_path).get_json("persist") == {"x": 1}
