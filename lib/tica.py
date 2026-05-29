import importlib.util
import os
import numpy as np
import torch
from mlcolvar.data import DictDataset
from mlcolvar.core.stats import TICA
from .reading import get_weights, set_flags_ACC_REJ, select_traj

def prepare_tica_dataset(
    pathensembles=None,
    lag_time=10,
    pcross=None,
    stride=1,
    return_info=False,
    trajectories=None,
    path_weights=None
):
    """
    Prepares a time-lagged dataset from TIS paths to be used with mlcolvar.


    Parameters
    ----------
    trajectories : list of np.ndarray, optional
        List of 2D arrays, expected shape (length_k, n_features), one trajectory per path.
    path_weights : list or np.ndarray, optional
        Weight of each path, one value per trajectory in `trajectories`.
    lag_time : int, optional
        The time lag to use for time-lagged pairs.
    pathensembles : list, optional
        List of PathEnsemble objects. If provided, `orderparameters` must also be given.
    orderparameters : list, optional
        List of OrderParameter objects aligned with `pathensembles`.
    pcross : list or np.ndarray, optional
        Ensemble-level scaling factors. Each ensemble's path weights are multiplied by
        the corresponding `pcross[i]`.

    Returns
    -------
    dataset : mlcolvar.data.DictDataset
        A DictDataset containing 'data', 'data_lag', 'weights', and 'weights_lag'.
    """
    if pathensembles is not None:
        if pathensembles[0].orders is None:
            raise ValueError("orderparameters must be provided when using pathensembles mode.")
        if pcross is None:
            pcross = np.ones(len(pathensembles), dtype=np.float32)
        else:
            pcross = np.asarray(pcross, dtype=np.float32)
            if len(pcross) != len(pathensembles):
                raise ValueError("pcross must have the same length as pathensembles.")

        trajectories = []
        path_weights = []
        for ensemble_index, pe in enumerate(pathensembles):
            ensemble_factor = float(pcross[ensemble_index])
            
            if hasattr(pe, "weights") and len(pe.weights) > 0:
                ensemble_path_weights = np.asarray(pe.weights, dtype=np.float32)
            else:
                ACCFLAGS, REJFLAGS = set_flags_ACC_REJ()
                ensemble_path_weights, _ = get_weights(pe.flags, ACCFLAGS, REJFLAGS, verbose=False)

            if len(ensemble_path_weights) != len(pe.lengths):
                raise ValueError(
                    "The number of weights in a PathEnsemble must match the number of paths."
                )

            for path_index in range(len(pe.lengths[:len(pe.orders)])):
                traj = pe.orders[path_index]
                trajectories.append(traj)
                path_weights.append(ensemble_path_weights[path_index] * ensemble_factor)

    if trajectories is None or path_weights is None:
        raise ValueError(
            "prepare_tica_dataset requires either trajectories/path_weights or pathensembles/orderparameters."
        )

    # 1. First pass: count total frames to pre-allocate PyTorch tensors
    valid_trajs = []
    valid_weights = []
    total_transitions = 0
    
    total_paths = len(trajectories)
    total_weight = np.sum(path_weights) if total_paths > 0 else 0.0
    
    for traj, w in zip(trajectories, path_weights):
        L_k = len(traj)
        if L_k <= lag_time:
            continue
        
        traj = np.asarray(traj, dtype=np.float32)
        if traj.ndim == 1:
            traj = traj.reshape(-1, 1)
            
        valid_trajs.append(traj)
        valid_weights.append(w)
        total_transitions += len(traj[:-lag_time:stride])

    if total_transitions == 0:
        raise ValueError("No trajectories are long enough for the given lag_time.")

    n_features = valid_trajs[0].shape[1]
    
    # 2. Pre-allocate tensors to avoid out-of-memory errors from data copies
    data = torch.empty((total_transitions, n_features), dtype=torch.float32)
    data_lag = torch.empty((total_transitions, n_features), dtype=torch.float32)
    weights = torch.empty((total_transitions,), dtype=torch.float32)
    
    # 3. Fill tensors in-place
    idx = 0
    for traj, w in zip(valid_trajs, valid_weights):
        x_t = traj[:-lag_time:stride]
        x_lag = traj[lag_time::stride]
        
        # Guard against off-by-one mismatches when striding
        n = len(x_t)
        if len(x_lag) > n:
            x_lag = x_lag[:n]
        elif len(x_lag) < n:
            x_t = x_t[:len(x_lag)]
            n = len(x_t)
            
        data[idx:idx+n] = torch.from_numpy(x_t)
        data_lag[idx:idx+n] = torch.from_numpy(x_lag)
        weights[idx:idx+n] = max(float(w), 1e-35)
        
        idx += n

    data_dict = {
        'data': data[:idx],
        'data_lag': data_lag[:idx],
        'weights': weights[:idx],
        'weights_lag': weights[:idx],
    }

    dataset = DictDataset(data_dict)
    
    if return_info:
        used_paths = len(valid_trajs)
        used_weight = np.sum(valid_weights) if used_paths > 0 else 0.0
        info = {
            'total_paths': total_paths,
            'used_paths': used_paths,
            'pct_paths_used': (used_paths / total_paths * 100) if total_paths > 0 else 0.0,
            'total_weight': total_weight,
            'used_weight': used_weight,
            'pct_weight_used': (used_weight / total_weight * 100) if total_weight > 0 else 0.0,
        }
        return dataset, info

    return dataset

def fit_mlcolvar_tica(dataset, n_cvs=1):
    """
    Initializes and fits an mlcolvar TICA model using the provided dataset.
    
    Parameters
    ----------
    dataset : mlcolvar.data.DictDataset
        The time-lagged dataset with weights.
    n_cvs : int
        Number of independent components to keep.
        
    Returns
    -------
    model : mlcolvar.core.stats.TICA
        The fitted TICA model.
    """
    in_features = dataset['data'].shape[1]
    
    # Initialize linear TICA model
    model = TICA(in_features=in_features, out_features=n_cvs)
    
    # Fit the TICA model by passing the time-lagged tensors as lists
    # to match mlcolvar's expected compute(data=[x_t, x_lag], weights=[w_t, w_lag]) API.
    ew, ev = model.compute(
        data=[dataset['data'], dataset['data_lag']],
        weights=[dataset['weights'], dataset['weights_lag']],
    )

    return ew, ev, model


import torch
import torch.nn as nn
import numpy as np
from mlcolvar.cvs import DeepTICA
from mlcolvar.data import DictDataset

class MultiTaskDeepTICA(DeepTICA):
    """
    Multi-Task Deep-TICA CV that jointly optimizes:
    1. TICA loss (maximizing sum of eigenvalues) on time-lagged pairs
    2. Turn-depth prediction loss on starting configurations of segments
    
    This helps the encoder learn slow modes that are also reactive.
    """
    def __init__(self, layers: list, n_cvs: int = None, options: dict = None, alpha: float = 0.5, loss_type: str = 'mse', warmup_steps: int = 0, **kwargs):
        super().__init__(layers, n_cvs, options, **kwargs)
        self.alpha = alpha
        self.warmup_steps = warmup_steps

        # Turn-depth head
        p = layers[-1]
        self.turn_depth_head = nn.Sequential(
            nn.Linear(p, 16),
            nn.Tanh(),
            nn.Linear(16, 1)
        )
        
        if loss_type == 'poisson':
            self.turn_depth_head.add_module('softplus', nn.Softplus())
            self.turn_depth_loss = nn.PoissonNLLLoss(log_input=False)
        elif loss_type == 'mse':
            self.turn_depth_loss = nn.MSELoss()
        else:
            raise ValueError(f"Unknown loss_type {loss_type}")

    def setup(self, stage=None):
        if stage == "fit":
            # Manually initialize norm_in to prevent keys mismatch from DictModule with DictDataset['dataset0']!
            if getattr(self, "norm_in", None) is not None and not getattr(self.norm_in, "is_initialized", False):
                if getattr(self.trainer, "datamodule", None) is not None:
                    stats = self.trainer.datamodule.train_dataloader().get_stats()
                elif getattr(self.trainer, "train_dataloader", None) is not None:
                    stats = self.trainer.train_dataloader.get_stats()
                else:
                    raise ValueError("Could not find datamodule or train_dataloader to initialize norm_in stats.")
                if "dataset0" in stats and "data" in stats["dataset0"]:
                    data_stats = stats["dataset0"]["data"]
                elif "data" in stats:
                    data_stats = stats["data"]
                else:
                    raise ValueError("Could not find 'data' stats for normalization in datamodule.")
                self.norm_in.set_from_stats(data_stats, self.norm_in.mode)
        
        # Then let parent setup proceed (which will skip norm_in since we just manually initialized it)
        super().setup(stage)

    def training_step(self, train_batch, batch_idx):
        """
        Expects train_batch to be a dict from DictModule containing multiple datasets:
        'dataset0' for TICA: dict with 'data', 'data_lag', 'weights', 'weights_lag'
        'dataset1' for Depth: dict with 'data' (start config) and 'turn_depth' (label)
        """
        loss_tica = torch.tensor(0.0, device=self.device)
        loss_depth = torch.tensor(0.0, device=self.device)
        eigvals = []
        eigvecs = None
        name = "train" if self.training else "valid"
        loss_dict = {}
        
        # --- 1. TICA task (dataset0) ---
        if "dataset0" in train_batch:
            batch_tica = train_batch["dataset0"]
            x_t = batch_tica["data"]
            x_lag = batch_tica["data_lag"]
            w_t = batch_tica.get("weights", torch.ones(x_t.shape[0], device=self.device))
            w_lag = batch_tica.get("weights_lag", torch.ones(x_lag.shape[0], device=self.device))
            
            f_t = self.forward_nn(x_t)
            f_lag = self.forward_nn(x_lag)
            
            eigvals_tica, eigvecs_tica = self.tica.compute(
                data=[f_t, f_lag], weights=[w_t, w_lag], save_params=True
            )
            loss_tica = self.loss_fn(eigvals_tica)
            eigvals = eigvals_tica
            eigvecs = eigvecs_tica
            loss_dict[f"{name}_loss_tica"] = loss_tica

            for i, eig in enumerate(eigvals_tica):
                loss_dict[f"{name}_eigval_{i+1}"] = eig
                
        # --- 2. Turn-depth task (dataset1) ---
        if "dataset1" in train_batch:
            batch_depth = train_batch["dataset1"]
            x_start = batch_depth["data"]
            label_depth = batch_depth["turn_depth"].view(-1, 1).float()
            
            f_start = self.forward_nn(x_start)
            
            # Sign correction for TICA CVs based on turn_depth
            if eigvecs is not None and "dataset0" in train_batch:
                # We project the representations onto the first eigenvector
                proj = torch.matmul(f_start, eigvecs[:, 0])
                
                # We want the CV to be positively correlated with depth.
                # Calculate covariance to ignore mean offsets.
                label_1d = label_depth.view(-1)
                cov = ((proj - proj.mean()) * (label_1d - label_1d.mean())).mean()

                if cov < 0:
                    with torch.no_grad():
                        # Flip the saved TICA eigenvector out-of-place to avoid
                        # inplace autograd corruption.
                        new_evecs = self.tica.evecs.clone()
                        new_evecs[:, 0] = -new_evecs[:, 0]
                        self.tica.evecs = new_evecs
            
            pred_depth = self.turn_depth_head(f_start)
            loss_depth = self.turn_depth_loss(pred_depth, label_depth)
            loss_dict[f"{name}_loss_depth"] = loss_depth
            
        
        # --- 3. Total Loss ---
        tica_weight = min(1.0, self.current_epoch / self.warmup_steps) if self.warmup_steps > 0 else 1.0
        
        loss = tica_weight * loss_tica + self.alpha * loss_depth
        loss_dict[f"{name}_loss"] = loss
        loss_dict[f"{name}_tica_weight"] = tica_weight
        
        self.log_dict(loss_dict, on_step=True, on_epoch=True)
        
        return loss

    def validation_step(self, val_batch, batch_idx):
        return self.training_step(val_batch, batch_idx)


def prepare_multitask_datasets(
    pathensembles=None,
    lag_time=10,
    pcross=None,
    stride=1,
    trajectories=None,
    path_weights=None
):
    """
    Prepares two datasets for Multi-Task DeepTICA.
    dataset0: TICA time-lagged pairs (like prepare_tica_dataset)
    dataset1: Turn-depth starting configurations and labels
    
    Returns
    -------
    dataset_tica : DictDataset
        Contains 'data', 'data_lag', 'weights', 'weights_lag'
    dataset_depth : DictDataset
        Contains 'data' (x_start), 'turn_depth' (D label)
    """
    # 1. Same logic to gather trajectories and weights
    if pathensembles is not None:
        trajectories = []
        path_weights = []
        turn_depths = []
        
        if pcross is None:
            pcross = np.ones(len(pathensembles), dtype=np.float32)
            
        for ensemble_index, pe in enumerate(pathensembles):
            ensemble_factor = float(pcross[ensemble_index])
            
            # Turn depth is the difference between lambmax and lambmin
            ensemble_turn_depths = (np.asarray(pe.lambmaxs, dtype=np.float32) - np.asarray(pe.lambmins, dtype=np.float32))
            
            if hasattr(pe, "weights") and len(pe.weights) > 0:
                ensemble_path_weights = np.asarray(pe.weights, dtype=np.float32)
            else:
                from .reading import set_flags_ACC_REJ, get_weights
                ACCFLAGS, REJFLAGS = set_flags_ACC_REJ()
                ensemble_path_weights, _ = get_weights(pe.flags, ACCFLAGS, REJFLAGS, verbose=False)

            for path_index in range(min(len(pe.orders), len(ensemble_path_weights))):
                traj = pe.orders[path_index]
                trajectories.append(traj)
                path_weights.append(ensemble_path_weights[path_index] * ensemble_factor)
                turn_depths.append(ensemble_turn_depths[path_index])
    else:
        raise ValueError("Currently only pathensembles mode supports extracting turn_depth automatically.")
        
    # 2. Build TICA dataset
    valid_trajs = []
    valid_weights = []
    total_transitions = 0
    
    start_configs = []
    start_depths = []
    
    for traj, w, d in zip(trajectories, path_weights, turn_depths):
        L_k = len(traj)
        if L_k <= lag_time:
            continue
            
        traj = np.asarray(traj, dtype=np.float32)
        if traj.ndim == 1:
            traj = traj.reshape(-1, 1)
            
        valid_trajs.append(traj)
        valid_weights.append(w)
        total_transitions += len(traj[:-lag_time:stride])
        
        # for dataset1: turn depth
        # start conf is the minimum of the path (turning point)
        turn_idx = np.argmin(traj[:, 0])
        start_configs.append(traj[turn_idx])
        start_depths.append(d)
        
    if total_transitions == 0:
        raise ValueError("No trajectories are long enough for the given lag_time.")
        
    n_features = valid_trajs[0].shape[1]
    
    # Pre-allocate TICA tensors
    data = torch.empty((total_transitions, n_features), dtype=torch.float32)
    data_lag = torch.empty((total_transitions, n_features), dtype=torch.float32)
    weights = torch.empty((total_transitions,), dtype=torch.float32)
    
    idx = 0
    for traj, w in zip(valid_trajs, valid_weights):
        x_t = traj[:-lag_time:stride]
        x_lag = traj[lag_time::stride]
        
        n = len(x_t)
        if len(x_lag) > n:
            x_lag = x_lag[:n]
        elif len(x_lag) < n:
            x_t = x_t[:len(x_lag)]
            n = len(x_t)
            
        data[idx:idx+n] = torch.from_numpy(x_t)
        data_lag[idx:idx+n] = torch.from_numpy(x_lag)
        weights[idx:idx+n] = max(float(w), 1e-35)
        
        idx += n
        
    dataset_tica = DictDataset({
        'data': data[:idx],
        'data_lag': data_lag[:idx],
        'weights': weights[:idx],
        'weights_lag': weights[:idx]
    })
    
    # Pre-allocate Depth tensors
    dataset_depth = DictDataset({
        'data': torch.tensor(np.array(start_configs), dtype=torch.float32),
        'turn_depth': torch.tensor(start_depths, dtype=torch.float32)
    })
    
    return dataset_tica, dataset_depth

