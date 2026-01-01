import numpy as np


# -----------------------------
# Referencing transforms
# -----------------------------

def apply_car(X: np.ndarray) -> np.ndarray:
    """Common average reference (CAR): subtract mean across channels."""
    if X.ndim == 3:
        return X - X.mean(axis=1, keepdims=True)
    if X.ndim == 2:
        return X - X.mean(axis=0, keepdims=True)
    raise ValueError(f"apply_car expects 2D or 3D array, got shape {getattr(X, 'shape', None)}")


def apply_ref_channel(X: np.ndarray, ref_idx: int) -> np.ndarray:
    """Monopolar re-reference to an existing recorded channel (e.g., Cz)."""
    ref_idx = int(ref_idx)
    if X.ndim == 3:
        return X - X[:, ref_idx : ref_idx + 1, :]
    if X.ndim == 2:
        return X - X[ref_idx : ref_idx + 1, :]
    raise ValueError(f"apply_ref_channel expects 2D or 3D array, got shape {getattr(X, 'shape', None)}")


def apply_laplacian(X: np.ndarray, neighbors: list[list[int]]) -> np.ndarray:
    """Local Laplacian-like spatial reference using a fixed neighbor list."""
    if neighbors is None:
        raise ValueError("neighbors must be provided for laplacian mode")

    if X.ndim == 3:
        N, C, T = X.shape
        if len(neighbors) != C:
            raise ValueError(f"neighbors length {len(neighbors)} must match channels {C}")
        out = X.copy()
        for i, nb in enumerate(neighbors):
            if not nb:
                continue
            out[:, i, :] = X[:, i, :] - X[:, nb, :].mean(axis=1)
        return out

    if X.ndim == 2:
        C, T = X.shape
        if len(neighbors) != C:
            raise ValueError(f"neighbors length {len(neighbors)} must match channels {C}")
        out = X.copy()
        for i, nb in enumerate(neighbors):
            if not nb:
                continue
            out[i, :] = X[i, :] - X[nb, :].mean(axis=0)
        return out

    raise ValueError(f"apply_laplacian expects 2D or 3D array, got shape {getattr(X, 'shape', None)}")


def apply_bipolar_nn(X: np.ndarray, neighbors: list[list[int]]) -> np.ndarray:
    """A simple bipolar-like transform: channel minus its first listed neighbor.

    Note: This is *not* a canonical clinical bipolar montage. It's a controlled,
    deterministic local-difference reference that keeps the same channel count
    (useful for invariance stress-tests).
    """
    if neighbors is None:
        raise ValueError("neighbors must be provided for bipolar mode")

    if X.ndim == 3:
        N, C, T = X.shape
        if len(neighbors) != C:
            raise ValueError(f"neighbors length {len(neighbors)} must match channels {C}")
        out = X.copy()
        for i, nb in enumerate(neighbors):
            if not nb:
                continue
            j = int(nb[0])
            out[:, i, :] = X[:, i, :] - X[:, j, :]
        return out

    if X.ndim == 2:
        C, T = X.shape
        if len(neighbors) != C:
            raise ValueError(f"neighbors length {len(neighbors)} must match channels {C}")
        out = X.copy()
        for i, nb in enumerate(neighbors):
            if not nb:
                continue
            j = int(nb[0])
            out[i, :] = X[i, :] - X[j, :]
        return out

    raise ValueError(f"apply_bipolar_nn expects 2D or 3D array, got shape {getattr(X, 'shape', None)}")


def apply_gram_schmidt_reference(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Gram-Schmidt style re-referencing.

    For each channel x_i, build a reference signal r_i as the mean of all other
    channels (excluding i), then remove the projection of x_i onto r_i:
        x_i' = x_i - <x_i, r_i>/<r_i, r_i> * r_i

    This avoids the trivial "one channel becomes identically zero" artifact that
    can happen with direct monopolar re-reference when the reference channel is
    kept as an input.
    """
    eps = float(eps)
    squeeze = False
    if X.ndim == 2:
        X = X[None, ...]
        squeeze = True
    if X.ndim != 3:
        raise ValueError(f"apply_gram_schmidt_reference expects 2D or 3D array, got shape {getattr(X, 'shape', None)}")

    Xf = np.asarray(X, dtype=np.float32)
    N, C, T = Xf.shape
    if C < 2:
        return Xf[0] if squeeze else Xf

    mean_all = Xf.mean(axis=1, keepdims=True)  # [N,1,T]
    out = np.empty_like(Xf)
    for i in range(C):
        # mean of other channels, excluding i
        ref = (mean_all[:, 0, :] * C - Xf[:, i, :]) / (C - 1)  # [N,T]
        num = (Xf[:, i, :] * ref).sum(axis=1)                  # [N]
        den = (ref * ref).sum(axis=1) + eps                    # [N]
        alpha = (num / den)[:, None]                           # [N,1]
        out[:, i, :] = Xf[:, i, :] - alpha * ref

    return out[0] if squeeze else out


def apply_reference(
    X: np.ndarray,
    mode: str = "native",
    ref_idx: int | None = None,
    lap_neighbors: list[list[int]] | None = None,
    drop_idx: int | None = None,
) -> np.ndarray:
    """Apply a referencing transform, optionally dropping one channel after."""
    mode = (mode or "native").strip().lower()

    if mode in ("native", "none", ""):
        out = X.astype(np.float32, copy=False)
    elif mode in ("car", "common_avg", "commonaverage", "average"):
        out = apply_car(X).astype(np.float32, copy=False)
    elif mode in ("ref", "cz_ref", "channel_ref"):
        if ref_idx is None:
            raise ValueError("ref_idx must be provided for ref mode")
        out = apply_ref_channel(X, ref_idx).astype(np.float32, copy=False)
    elif mode in ("laplacian", "lap", "local"):
        out = apply_laplacian(X, lap_neighbors).astype(np.float32, copy=False)
    elif mode in ("bipolar", "bip", "bipolar_nn"):
        out = apply_bipolar_nn(X, lap_neighbors).astype(np.float32, copy=False)
    elif mode in ("gs", "gram_schmidt", "gram-schmidt"):
        out = apply_gram_schmidt_reference(X).astype(np.float32, copy=False)
    else:
        raise ValueError(f"Unknown reference mode: {mode}")

    if drop_idx is not None:
        di = int(drop_idx)
        if out.ndim == 3:
            out = np.delete(out, di, axis=1)
        elif out.ndim == 2:
            out = np.delete(out, di, axis=0)
        else:
            raise ValueError(f"apply_reference expects 2D or 3D output, got shape {getattr(out, 'shape', None)}")

    return out


# -----------------------------
# Euclidean Alignment (EA)
# -----------------------------

def _eigh_inv_sqrt(M: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Stable inverse square-root for symmetric (PSD-ish) matrices."""
    M = np.asarray(M, dtype=np.float64)
    M = 0.5 * (M + M.T)  # enforce symmetry
    w, v = np.linalg.eigh(M)
    w = np.clip(w, a_min=float(eps), a_max=None)
    return v @ np.diag(1.0 / np.sqrt(w)) @ v.T


def ea_align_trials(X: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Euclidean Alignment (EA) over a set of trials.

    X: [N,C,T]
    Returns: [N,C,T] after left-multiplying each trial by Rbar^{-1/2}, where
    Rbar is the mean covariance across trials.

    The implementation is intentionally defensive: it symmetrizes covariances,
    adds diagonal jitter, and clips eigenvalues to avoid NaNs.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 3:
        raise ValueError(f"ea_align_trials expects [N,C,T], got shape {getattr(X, 'shape', None)}")

    N, C, T = X.shape
    if N == 0 or C == 0 or T == 0:
        return X.astype(np.float32, copy=False)

    cov_sum = np.zeros((C, C), dtype=np.float64)
    for n in range(N):
        xn = X[n]
        Rn = (xn @ xn.T) / max(1, T)
        Rn = 0.5 * (Rn + Rn.T)
        cov_sum += Rn

    Rbar = cov_sum / max(1, N)
    Rbar = 0.5 * (Rbar + Rbar.T)
    Rbar += float(eps) * np.eye(C, dtype=np.float64)

    W = _eigh_inv_sqrt(Rbar, eps=eps)

    Xea = np.empty((N, C, T), dtype=np.float32)
    for n in range(N):
        Xea[n] = (W @ X[n]).astype(np.float32)

    return Xea
