########################################################################################################################
# GPU-accelerated batched WLS regression
#
# SCOPE: this GPU path only handles method == 'Linear' AND dynamic_predictors['flag'] == False
#
# Logistic regression and the dynamic-predictor path are NOT handled here. Both fall back to the existing
# CPU multiprocessing path completely unchanged. Batching either of those cleanly is a much bigger
# undertaking and was scoped out to keep this addition simple
#
# The math is identical to the CPU precompute path already validated against statsmodels WLS:
#   b = (X^T W X)^{-1} X^T W y
#   yhat = x_g @ b = [x_g @ (X^T W X)^{-1} X^T W] @ y = projector @ y
# Because X, W, x_g are static across time when dynamic predictors are off, `projector` is computed once per
# grid cell and reused across all ntime days
#
# tar_nearIndex / tar_nearWeight are already padded to a fixed `nearmax` per cell (invalid neighbors marked
# with index < 0), so no ragged-array handling is needed, only a validity mask.
########################################################################################################################

import time
import numpy as np


def check_gpu_available():
    """Return True if a usable CUDA GPU + CuPy are available, else False."""
    try:
        import cupy as cp
        # Force a trivial allocation/op to confirm the device actually works,
        # not just that the import succeeded.
        _ = cp.zeros(1) + 1
        cp.cuda.Stream.null.synchronize()
        return True
    except Exception:
        return False


def regression_grid_gpu_linear_static(stn_data, stn_predictor, tar_nearIndex, tar_nearWeight, tar_predictor,
                                       cell_batch_size=20000, time_chunk_size=365, dtype='float64', ridge_eps=1e-12):
    """
    GPU-batched WLS regression for the entire grid, for method == 'Linear' with dynamic predictors off.

    Parameters
    ----------
    stn_data : (nstn, ntime) ndarray
        Station observations.
    stn_predictor : (nstn, npred) ndarray
        Static predictors at station locations.
    tar_nearIndex : (nrow, ncol, nearmax) ndarray (int)
        Index of neighboring stations per grid cell; -1 marks an unused/padding slot.
    tar_nearWeight : (nrow, ncol, nearmax) ndarray
        Distance weights of neighboring stations per grid cell.
    tar_predictor : (nrow, ncol, npred) ndarray
        Static predictors at grid cell locations.
    cell_batch_size : int
        Number of grid cells processed per GPU batch. Tune down if you hit out-of-memory errors.
    time_chunk_size : int
        Number of timesteps gathered to GPU at once per cell batch. Tune down if you hit out-of-memory errors.
    dtype : str
        'float64' (matches CPU/statsmodels precision) or 'float32'
    ridge_eps : float
        Tiny value added to the diagonal of (X^T W X) before solving, scaled by each matrix's own trace.
        This prevents a single near-singular cell from poisoning/erroring out an entire batched solve.
        True singular cells are still detected via determinant and forced to NaN afterward (matching the
        original CPU behavior), so this does not change results for well-conditioned cells.

    Returns
    -------
    estimates : (nrow, ncol, ntime) ndarray (float32)
    """
    import cupy as cp

    t1 = time.time()
    np_dtype = np.float64 if dtype == 'float64' else np.float32

    nstn, ntime = np.shape(stn_data)
    nrow, ncol, nearmax = np.shape(tar_nearIndex)
    npred = tar_predictor.shape[-1]

    estimates = np.nan * np.zeros([nrow, ncol, ntime], dtype=np.float32)

    # Flatten the grid and keep only cells with at least one valid neighbor
    near_index_flat  = tar_nearIndex.reshape(nrow * ncol, nearmax)
    near_weight_flat = tar_nearWeight.reshape(nrow * ncol, nearmax)
    tar_pred_flat    = tar_predictor.reshape(nrow * ncol, npred)

    valid_mask_per_cell = near_index_flat >= 0
    n_valid_neighbors    = valid_mask_per_cell.sum(axis=1)
    valid_cells          = np.where(n_valid_neighbors > 0)[0]   # flat indices into nrow*ncol

    if len(valid_cells) == 0:
        return np.squeeze(estimates)

    # Move station data to GPU once; it is reused by every cell batch.
    stn_data_gpu      = cp.asarray(stn_data, dtype=np_dtype)        # (nstn, ntime)
    stn_predictor_gpu = cp.asarray(stn_predictor, dtype=np_dtype)   # (nstn, npred)

    n_batches = int(np.ceil(len(valid_cells) / cell_batch_size))

    predictor_global_var = cp.asarray(stn_predictor, dtype=np_dtype).var(axis=0)   # (npred,)

    for b in range(n_batches):
        batch_cells = valid_cells[b * cell_batch_size: (b + 1) * cell_batch_size]
        B = len(batch_cells)

        # --- gather per-cell neighbor indices/weights/predictors, with -1 -> 0 + zero weight for padding ---
        idx_batch    = near_index_flat[batch_cells, :].copy()        # (B, nearmax)
        weight_batch = near_weight_flat[batch_cells, :].copy()       # (B, nearmax)
        valid_batch  = idx_batch >= 0                                # (B, nearmax)

        idx_batch[~valid_batch] = 0          # safe placeholder index (weight is zeroed below so it's inert)
        weight_batch[~valid_batch] = 0.0

        idx_gpu    = cp.asarray(idx_batch)                                   # (B, nearmax) int
        weight_gpu = cp.asarray(weight_batch, dtype=np_dtype)                # (B, nearmax)
        valid_gpu  = cp.asarray(valid_batch)                                 # (B, nearmax) bool

        xg_gpu = cp.asarray(tar_pred_flat[batch_cells, :], dtype=np_dtype)   # (B, npred)

        # X: (B, nearmax, npred) -- gather predictor rows for each neighbor of each cell
        X = stn_predictor_gpu[idx_gpu]              # fancy indexing: (B, nearmax, npred)
        X = X * valid_gpu[:, :, None]                # zero out padded rows

        # --- build and solve the WLS normal equations for the whole batch at once ---
        # XtW: (B, npred, nearmax) = X^T * W  (W applied along the neighbor axis)
        XtW  = cp.transpose(X, (0, 2, 1)) * weight_gpu[:, None, :]
        XtWX = cp.matmul(XtW, X)                      # (B, npred, npred)

        n_valid = valid_gpu.sum(axis=1)                      # (B,)

        # Flag any predictor column (excluding col 0, constant) that has ~zero variance
        # across a cell's *valid* neighbors. 
        # Variance must be computed over valid entries only
        mean = cp.einsum('bnp->bp', X) / n_valid[:, None]                      # (B, npred)
        diff = (X - mean[:, None, :]) * valid_gpu[:, :, None]                  # zero out invalid rows
        var  = cp.einsum('bnp,bnp->bp', diff, diff) / n_valid[:, None]         # (B, npred)

        # Scale the near-zero threshold relative to each predictor's *global* variance
        # across all stations
        rel_var_eps = 1e-8
        near_zero_var = var[:, 1:] < (rel_var_eps * predictor_global_var[None, 1:])   # skip intercept col
        collinear = cp.any(near_zero_var, axis=1)                                     # (B,)

        singular = (n_valid < npred) | collinear

        trace = cp.einsum('bii->b', XtWX)
        eye = cp.eye(npred, dtype=np_dtype)[None, :, :]
        XtWX_reg = XtWX + ridge_eps * (trace[:, None, None] + 1.0) * eye

        Z = cp.linalg.solve(XtWX_reg, XtW)
        projector = cp.einsum('bp,bpm->bm', xg_gpu, Z)
        projector = projector * valid_gpu
        projector[singular, :] = cp.nan

        # --- apply the projector across all timesteps, in time chunks to bound GPU memory ---
        n_time_batches = int(np.ceil(ntime / time_chunk_size))
        batch_result = cp.empty((B, ntime), dtype=np_dtype)

        for tb in range(n_time_batches):
            t0 = tb * time_chunk_size
            t1c = min(ntime, t0 + time_chunk_size)

            # Y_near: (B, nearmax, time_chunk)
            Y_near = stn_data_gpu[idx_gpu, t0:t1c]
            Y_near = cp.where(valid_gpu[:, :, None], Y_near, cp.nan)

            # vectorized version of: "if all neighbor values are identical, use that value directly"
            y_min = cp.nanmin(Y_near, axis=1)         # (B, time_chunk)
            y_max = cp.nanmax(Y_near, axis=1)
            all_equal = (y_min == y_max)

            Y_near_filled = cp.where(cp.isnan(Y_near), 0.0, Y_near)
            regressed = cp.einsum('bm,bmt->bt', projector, Y_near_filled)   # (B, time_chunk)

            batch_result[:, t0:t1c] = cp.where(all_equal, y_min, regressed)

        # --- bring batch back to host and scatter into the output grid ---
        batch_result_cpu = cp.asnumpy(batch_result).astype(np.float32)
        rows = batch_cells // ncol
        cols = batch_cells % ncol
        estimates[rows, cols, :] = batch_result_cpu

    t2 = time.time()
    print(f'GPU regression time cost (sec): {t2 - t1:.2f}  ({len(valid_cells)} cells, {ntime} timesteps)')

    return np.squeeze(estimates)


def loop_regression_2Dor3D_auto(stn_data, stn_predictor, tar_nearIndex, tar_nearWeight, tar_predictor, method, probflag,
                                settings, dynamic_predictors={}, num_processes=4, importmodules=[], maxlimit={},
                                gpu_cell_batch_size=20000, gpu_time_chunk_size=365, gpu_dtype='float64', force_cpu=False):
    """
    Drop-in replacement for loop_regression_2Dor3D_multiprocessing that automatically uses the GPU batched
    path when:
      - a usable GPU + CuPy is available,
      - method == 'Linear',
      - dynamic_predictors['flag'] == False (or dynamic_predictors == {}),
      - force_cpu is not set.

    In every other case (Logistic, any ML method, dynamic predictors on, or no GPU available), this calls
    the existing, unmodified loop_regression_2Dor3D_multiprocessing
    """
    # import here so a missing `regression` module during standalone testing of this file doesn't error
    from regression import loop_regression_2Dor3D_multiprocessing

    if len(dynamic_predictors) == 0:
        dynamic_predictors = {'flag': False}

    use_gpu = (not force_cpu) and (method == 'Linear') and (dynamic_predictors.get('flag', False) == False)

    if use_gpu:
        gpu_ok = check_gpu_available()
        if gpu_ok:
            print('GPU available: using batched GPU regression for static-predictor Linear case.')
            return regression_grid_gpu_linear_static(
                stn_data, stn_predictor, tar_nearIndex, tar_nearWeight, tar_predictor,
                cell_batch_size=gpu_cell_batch_size, time_chunk_size=gpu_time_chunk_size, dtype=gpu_dtype)
        else:
            print('No usable GPU found: falling back to CPU multiprocessing path.')

    return loop_regression_2Dor3D_multiprocessing(
        stn_data, stn_predictor, tar_nearIndex, tar_nearWeight, tar_predictor, method, probflag,
        settings, dynamic_predictors=dynamic_predictors, num_processes=num_processes,
        importmodules=importmodules, maxlimit=maxlimit)
