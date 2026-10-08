import json

import numpy as np
from sklearn.metrics import adjusted_rand_score

from taxi.worker import run


def test_metrics_and_nested_data(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    (tmp_path / "artifacts").mkdir()
    rng = np.random.default_rng(42)
    x = np.concatenate([rng.normal(-5, 0.1, (100, 6)), rng.normal(5, 0.1, (100, 6))]).astype(
        "float32"
    )
    order = rng.permutation(len(x))
    np.save("data/benchmark.npy", x)
    np.save("data/benchmark_order.npy", order)
    ordinary = run("KMeans", 200, 2, 100, False)
    mini = run("MiniBatchKMeans", 200, 2, 100, False)
    streamed = run("MiniBatchKMeans", 200, 2, 100, True)
    for result in [ordinary, mini, streamed]:
        assert result["status"] == "ok"
        assert sum(json.loads(result["cluster_sizes"])) == 200
        assert 0 < result["normalized_inertia"] < 0.1
        assert result["silhouette"] > 0.9
        assert result["fit_seconds"] > 0
    assert mini["ari_vs_kmeans"] == 1
    assert adjusted_rand_score([0, 0, 1, 1], [1, 1, 0, 0]) == 1
