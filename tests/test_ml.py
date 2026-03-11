import pytest
import numpy as np
from quasar.supervised import SupervisedModels
from quasar.unsupervised import UnsupervisedModels

def test_kmeans_separates_blobs():
    # Two distinct blobs
    X = [
        [0, 0], [0.1, 0.1], [0, 0.1],
        [10, 10], [10.1, 10.1], [10, 10.1]
    ]

    result = UnsupervisedModels.train_kmeans(X, n_clusters=2)

    labels = result['labels']
    assert len(labels) == 6
    # The first 3 should have one label, the last 3 another
    assert labels[0] == labels[1] == labels[2]
    assert labels[3] == labels[4] == labels[5]
    assert labels[0] != labels[3]

    centers = result['cluster_centers']
    assert len(centers) == 2

def test_svm_identifies_support_vectors():
    X = [
        [1, 1], [2, 2], [2, 0],
        [8, 8], [9, 9], [9, 7]
    ]
    y = [0, 0, 0, 1, 1, 1]

    result = SupervisedModels.train_svm(X, y, kernel='linear', C=1.0)

    sv = result['support_vectors']
    assert len(sv) > 0
    # Check if support vectors are close to the boundary (2,2 and 8,8 for example)
    sv_list = [list(v) for v in sv]
    assert [2.0, 2.0] in sv_list or [8.0, 8.0] in sv_list
