import numpy as np
from typing import List, Union


def cosine_similarity(vec1: Union[List[float], np.ndarray], vec2: Union[List[float], np.ndarray]) -> float:
    if not isinstance(vec1, np.ndarray):
        vec1 = np.array(vec1)
    if not isinstance(vec2, np.ndarray):
        vec2 = np.array(vec2)

    if vec1.size == 0 or vec2.size == 0:
        return 0.0

    if vec1.shape != vec2.shape:
        raise ValueError(f"Vectors must have same shape: {vec1.shape} vs {vec2.shape}")

    dot_product = np.dot(vec1, vec2)

    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    similarity = dot_product / (norm1 * norm2)

    similarity = np.clip(similarity, -1.0, 1.0)

    return float(similarity)
