import abc
import logging
from pathlib import Path
from typing import TypeVar, Generic, cast, Any

import numpy as np
import numpy.typing as npt
import torch
import faiss
from faiss import Index

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=Index)
NumpyArray = npt.NDArray[np.float32]


class FaissFeatureIndex(Generic[T], abc.ABC):
    def __init__(self, index: T) -> None:
        self._index = index

    def save(self, filepath: Path, rewrite: bool = False) -> None:
        if filepath.exists() and not rewrite:
            raise FileExistsError(f"index already exists by path {filepath}")
        faiss.write_index(self._index, str(filepath))


class FaissRetrievableFeatureIndex(FaissFeatureIndex[Index], abc.ABC):
    """retrieve voice feature vectors by faiss index"""

    def __init__(self, index: T, ratio: float, n_nearest_vectors: int) -> None:
        super().__init__(index=index)
        if index.metric_type != self.supported_distance:
            raise ValueError(f"index metric type {index.metric_type=} is unsupported {self.supported_distance=}")

        if 1 > n_nearest_vectors:
            raise ValueError("n-retrieval-vectors must be gte 1")
        self._n_nearest = n_nearest_vectors

        if 0 > ratio > 1:
            raise ValueError(f"{ratio=} must be in rage (0, 1)")
        self._ratio = ratio

    @property
    @abc.abstractmethod
    def supported_distance(self) -> Any:
        raise NotImplementedError

    @abc.abstractmethod
    def _weight_nearest_vectors(self, nearest_vectors: NumpyArray, scores: NumpyArray) -> NumpyArray:
        raise NotImplementedError

    def retriv(self, features: NumpyArray) -> NumpyArray:
        # use method search_and_reconstruct instead of recreating the whole matrix
        scores, _, nearest_vectors = self._index.search_and_reconstruct(features, k=self._n_nearest)
        weighted_nearest_vectors = self._weight_nearest_vectors(nearest_vectors, scores)
        retriv_vector = (1 - self._ratio) * features + self._ratio * weighted_nearest_vectors
        return retriv_vector
    

class FaissRVCRetrievableFeatureIndex(FaissRetrievableFeatureIndex):
    """
    retrieve voice encoded features with algorith from RVC repository
    https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI
    """

    @property
    def supported_distance(self) -> Any:
        return faiss.METRIC_L2

    def _weight_nearest_vectors(self, nearest_vectors: NumpyArray, scores: NumpyArray) -> NumpyArray:
        """
        magic code from original RVC
        https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/86ed98aacaa8b2037aad795abd11cdca122cf39f/vc_infer_pipeline.py#L213C18-L213C19

        nearest_vectors dim (n_nearest, vector_dim)
        scores dim (num_vectors, n_nearest)
        """
        logger.debug("shape: nv=%s sc=%s", nearest_vectors.shape, scores.shape)
        weight = np.square(1 / scores)
        weight /= weight.sum(axis=1, keepdims=True)
        weight = np.expand_dims(weight, axis=2)
        weighted_nearest_vectors = np.sum(nearest_vectors * weight, axis=1)
        logger.debug(
            "shape: nv=%s weight=%s weight_nearest=%s",
            nearest_vectors.shape,
            weight.shape,
            weighted_nearest_vectors.shape,
        )
        return cast(NumpyArray, weighted_nearest_vectors)
    

class IRetrieval(abc.ABC):
    @abc.abstractmethod
    def retriv_whisper(self, vec: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abc.abstractmethod
    def retriv_hubert(self, vec: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class DummyRetrieval(IRetrieval):
    def retriv_whisper(self, vec: torch.FloatTensor) -> torch.FloatTensor:
        logger.debug("start dummy retriv whisper")
        return vec.clone().to(torch.device("cpu"))

    def retriv_hubert(self, vec: torch.FloatTensor) -> torch.FloatTensor:
        logger.debug("start dummy retriv hubert")
        return vec.clone().to(torch.device("cpu"))


class FaissIndexRetrieval(IRetrieval):
    def __init__(self, hubert_index: FaissRetrievableFeatureIndex, whisper_index: FaissRetrievableFeatureIndex) -> None:
        self._hubert_index = hubert_index
        self._whisper_index = whisper_index

    def retriv_whisper(self, vec: torch.Tensor) -> torch.Tensor:
        logger.debug("start retriv whisper")
        np_vec = self._whisper_index.retriv(vec.numpy())
        return torch.from_numpy(np_vec)

    def retriv_hubert(self, vec: torch.Tensor) -> torch.Tensor:
        logger.debug("start retriv hubert")
        np_vec = self._hubert_index.retriv(vec.numpy())
        return torch.from_numpy(np_vec)
    
def load_retrieve_index(filepath: Path, ratio: float, n_nearest_vectors: int) -> FaissRetrievableFeatureIndex:
    return FaissRVCRetrievableFeatureIndex(
        index=faiss.read_index(str(filepath)), ratio=ratio, n_nearest_vectors=n_nearest_vectors
    )