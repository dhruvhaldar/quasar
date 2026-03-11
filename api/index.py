from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import os
import sys

# Ensure quasar module can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quasar.supervised import SupervisedModels
from quasar.neural_networks import NeuralNetworkModels
from quasar.unsupervised import UnsupervisedModels

app = FastAPI(title="Quasar API", description="API for modern statistical learning workbench.")

class DataPayload(BaseModel):
    X: List[List[float]]
    y: Optional[List[int]] = None

class SVMRequest(DataPayload):
    kernel: str = 'linear'
    C: float = 1.0
    cv: bool = False

class KmeansRequest(DataPayload):
    n_clusters: int = 3

class ANNRequest(DataPayload):
    hidden_size: int = 10
    epochs: int = 100
    learning_rate: float = 0.01

@app.post("/api/train/svm")
def train_svm(request: SVMRequest):
    result = SupervisedModels.train_svm(request.X, request.y, request.kernel, request.C, cv=request.cv)
    return result

@app.post("/api/train/kmeans")
def train_kmeans(request: KmeansRequest):
    result = UnsupervisedModels.train_kmeans(request.X, request.n_clusters)
    return result

@app.post("/api/train/ann")
def train_ann(request: ANNRequest):
    result = NeuralNetworkModels.train_ann(request.X, request.y, request.hidden_size, request.epochs, request.learning_rate)
    return result

# Vercel requires app to be available in this module
