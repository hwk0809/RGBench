import numpy as np
from typing import Union

import torch
from scipy.spatial.distance import cdist

# PyKeOps is an optional GPU-acceleration dependency. We detect it once at
# import time so the auto-dispatching ``chamfer_distance`` below can fall
# back to the SciPy CPU path without per-call exception handling.
try:
    import pykeops.torch  # noqa: F401
    _PYKEOPS_AVAILABLE = True
except Exception:
    _PYKEOPS_AVAILABLE = False


def chamfer_distance_single_direction_cpu(p1_np: np.ndarray, p2_np: np.ndarray, norm: int) -> float:
    """
    Computes single-directional Chamfer Distance using SciPy.
    This runs on the CPU.

    Args:
        p1_np: Source point cloud as a NumPy array (N, D).
        p2_np: Target point cloud as a NumPy array (M, D).
        norm: 1 for L1 (Manhattan), 2 for L2 (Euclidean).

    Returns:
        The mean Chamfer distance.
    """
    if norm == 1:
        metric = 'cityblock'
    elif norm == 2:
        metric = 'euclidean'
    else:
        raise ValueError("SciPy implementation only supports norm 1 or 2.")

    dist_matrix = cdist(p1_np, p2_np, metric=metric)
    min_dists = np.min(dist_matrix, axis=1)
    return np.mean(min_dists)


def chamfer_distance_single_direction_gpu(p1_torch: torch.Tensor, p2_torch: torch.Tensor, norm: int) -> torch.Tensor:
    """
    Computes single-directional Chamfer Distance using PyKeOps.
    Handles batches and runs on GPU if available.

    Args:
        p1_torch: Source point cloud as a PyTorch tensor (B, N, D).
        p2_torch: Target point cloud as a PyTorch tensor (B, M, D).
        norm: 1 for L1 (Manhattan), 2 for L2 (Euclidean).

    Returns:
        A tensor containing the mean Chamfer distance over the batch.
    """
    # pykeops is an optional dependency; defer the import so the CPU
    # chamfer path keeps working when it's not installed.
    from pykeops.torch import LazyTensor

    # Ensure tensors have a batch dimension
    if p1_torch.ndim == 2:
        p1_torch = p1_torch.unsqueeze(0)
    if p2_torch.ndim == 2:
        p2_torch = p2_torch.unsqueeze(0)

    # Convert to LazyTensors for memory-efficient computation
    p1_lazy = LazyTensor(p1_torch.unsqueeze(2))  # (B, N, 1, D)
    p2_lazy = LazyTensor(p2_torch.unsqueeze(1))  # (B, 1, M, D)

    if norm == 1:
        # L1 distance
        dist_matrix = (p1_lazy - p2_lazy).abs().sum(-1)
    elif norm == 2:
        # L2 distance
        dist_matrix_sq = ((p1_lazy - p2_lazy) ** 2).sum(-1)
        dist_matrix = dist_matrix_sq.sqrt()
    else:
        raise ValueError("PyKeOps implementation only supports norm 1 or 2.")

    # Find nearest neighbor distance for each point in p1
    min_dists_per_point = dist_matrix.min(dim=2)  # Shape: (B, N)

    # Average distances per point cloud (point_reduction='mean')
    mean_dist_per_cloud = min_dists_per_point.mean(dim=1)  # Shape: (B,)

    # Average over the batch (batch_reduction='mean')
    final_mean_dist = mean_dist_per_cloud.mean()

    return final_mean_dist


def chamfer_distance(p1: Union[np.ndarray, torch.Tensor],
                     p2: Union[np.ndarray, torch.Tensor],
                     norm: int) -> float:
    """Dispatch one-directional Chamfer to the best available backend.

    Uses the PyKeOps + CUDA GPU path when both are available, otherwise
    falls back to the SciPy CPU path. Accepts NumPy arrays or PyTorch
    tensors of shape ``(N, D)`` or ``(B, N, D)``; returns a Python float
    in either case so callers don't need to care about backend.
    """
    use_gpu = _PYKEOPS_AVAILABLE and torch.cuda.is_available()

    if use_gpu:
        if isinstance(p1, np.ndarray):
            p1 = torch.from_numpy(np.asarray(p1, dtype=np.float32))
        if isinstance(p2, np.ndarray):
            p2 = torch.from_numpy(np.asarray(p2, dtype=np.float32))
        out = chamfer_distance_single_direction_gpu(p1.float(), p2.float(), norm)
        return float(out.detach().cpu())

    # CPU path expects 2D (N, D) numpy arrays.
    if isinstance(p1, torch.Tensor):
        p1 = p1.detach().cpu().numpy()
    if isinstance(p2, torch.Tensor):
        p2 = p2.detach().cpu().numpy()
    p1 = np.asarray(p1, dtype=np.float32)
    p2 = np.asarray(p2, dtype=np.float32)
    if p1.ndim == 3:
        p1 = p1[0]
    if p2.ndim == 3:
        p2 = p2[0]
    return float(chamfer_distance_single_direction_cpu(p1, p2, norm))


def calculate_cloth_sim_stability(vertices: np.ndarray) -> float:
    """
    Calculates the simulation stability metric (L_s) for a sequence of vertices.

    This metric measures the smoothness of the vertex sequence by calculating how much
    each vertex deviates from the average of its local neighborhood. A lower value
    indicates a more stable or smoother sequence.

    The formula is: L_s = (1/N) * Σ |(v_{t-1} + v_t + v_{t+1})/3 - v_t|

    Args:
        vertices (np.ndarray): A NumPy array of shape (N, D), where N is the number
                             of vertices and D is the dimension (e.g., 3 for 3D).
                             The vertices are assumed to be in a meaningful order.

    Returns:
        float: The calculated stability metric (L_s).
    """
    # Get the total number of vertices, N.
    num_vertices = vertices.shape[0]

    # The formula requires a window of 3 vertices, so we need at least 3.
    # If not, there is no deviation to measure.
    if num_vertices < 3:
        return 0.0

    # Use NumPy slicing for an efficient, vectorized calculation.
    # This avoids a slow Python `for` loop.
    # v_t in the formula corresponds to the central vertices in each 3-point window.
    v_t = vertices[1:-1]
    # v_{t-1} are the vertices preceding v_t.
    v_t_minus_1 = vertices[:-2]
    # v_{t+1} are the vertices succeeding v_t.
    v_t_plus_1 = vertices[2:]

    # Calculate the smoothed vertex position (the local average).
    smoothed_v = (v_t_minus_1 + v_t + v_t_plus_1) / 3.0

    # Calculate the difference vector between the smoothed and original position.
    diff_vectors = smoothed_v - v_t

    # The absolute value |...| on a vector denotes its magnitude (L2-norm).
    # We calculate the L2-norm for each difference vector along the coordinate axis (axis=1).
    magnitudes = np.linalg.norm(diff_vectors, axis=1)

    # Sum the magnitudes as per the formula's summation part.
    total_deviation = np.sum(magnitudes)

    # Average the total deviation by N, the total number of vertices.
    stability_metric = total_deviation / num_vertices

    return stability_metric