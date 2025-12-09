"""
Common Utilities for Quote Attribution Training

Shared utilities used across all training modules:
- Checkpoint management
- Logging configuration
- Device setup
- Metrics computation
- Progress tracking
"""

import os
import json
import time
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def setup_logging(
    log_dir: str = "logs",
    log_file: Optional[str] = None,
    level: int = logging.INFO
) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        log_dir: Directory for log files
        log_file: Optional specific log file name
        level: Logging level
        
    Returns:
        Configured logger
    """
    os.makedirs(log_dir, exist_ok=True)
    
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"training_{timestamp}.log"
    
    log_path = os.path.join(log_dir, log_file)
    
    # CURSOR: Configure logging
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging to {log_path}")
    
    return logger


def get_device(prefer_cuda: bool = True) -> torch.device:
    """
    Get the best available device.
    
    Args:
        prefer_cuda: Whether to prefer CUDA over CPU
        
    Returns:
        torch.device
    """
    if prefer_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    return device


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count model parameters.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with parameter counts
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total,
        'trainable': trainable,
        'frozen': total - trainable,
        'total_mb': total * 4 / (1024 ** 2),  # Assuming float32
    }


class CheckpointManager:
    """
    Manages model checkpoints with automatic cleanup.
    """
    
    def __init__(
        self,
        checkpoint_dir: str,
        max_checkpoints: int = 5,
        save_best_only: bool = False
    ):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory for checkpoints
            max_checkpoints: Maximum checkpoints to keep
            save_best_only: Only save when metric improves
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.save_best_only = save_best_only
        
        self.best_metric = float('-inf')
        self.checkpoints: List[Tuple[str, float]] = []
    
    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        step: int,
        metric: float,
        additional_info: Optional[Dict] = None
    ) -> Optional[str]:
        """
        Save a checkpoint.
        
        Args:
            model: Model to save
            optimizer: Optimizer state
            epoch: Current epoch
            step: Current step
            metric: Current metric value
            additional_info: Additional info to save
            
        Returns:
            Checkpoint path if saved, None otherwise
        """
        # CURSOR: Check if we should save
        if self.save_best_only and metric <= self.best_metric:
            return None
        
        # CURSOR: Create checkpoint
        checkpoint = {
            'epoch': epoch,
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metric': metric,
            'timestamp': datetime.now().isoformat()
        }
        
        if additional_info:
            checkpoint.update(additional_info)
        
        # CURSOR: Save checkpoint
        if metric > self.best_metric:
            self.best_metric = metric
            filename = f"best_model.pt"
        else:
            filename = f"checkpoint_step_{step}.pt"
        
        filepath = self.checkpoint_dir / filename
        torch.save(checkpoint, filepath)
        
        # CURSOR: Track checkpoints
        self.checkpoints.append((str(filepath), metric))
        
        # CURSOR: Cleanup old checkpoints
        self._cleanup()
        
        return str(filepath)
    
    def _cleanup(self):
        """Remove old checkpoints if exceeding max."""
        # Keep best model separate
        regular_checkpoints = [
            (p, m) for p, m in self.checkpoints
            if 'best_model' not in p
        ]
        
        while len(regular_checkpoints) > self.max_checkpoints - 1:
            oldest_path, _ = regular_checkpoints.pop(0)
            if os.path.exists(oldest_path):
                os.remove(oldest_path)
    
    def load_best(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: str = 'cpu'
    ) -> Dict[str, Any]:
        """
        Load the best checkpoint.
        
        Args:
            model: Model to load into
            optimizer: Optional optimizer to restore
            device: Device to load to
            
        Returns:
            Checkpoint info dictionary
        """
        best_path = self.checkpoint_dir / "best_model.pt"
        return self.load(best_path, model, optimizer, device)
    
    def load(
        self,
        checkpoint_path: str,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: str = 'cpu'
    ) -> Dict[str, Any]:
        """
        Load a specific checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint
            model: Model to load into
            optimizer: Optional optimizer to restore
            device: Device to load to
            
        Returns:
            Checkpoint info dictionary
        """
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return {
            'epoch': checkpoint.get('epoch', 0),
            'step': checkpoint.get('step', 0),
            'metric': checkpoint.get('metric', 0),
            'timestamp': checkpoint.get('timestamp', '')
        }
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """Get path to most recent checkpoint."""
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_step_*.pt"))
        if not checkpoints:
            return None
        
        # Sort by step number
        checkpoints.sort(key=lambda p: int(p.stem.split('_')[-1]))
        return str(checkpoints[-1])


def compute_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    average: str = 'weighted'
) -> Dict[str, float]:
    """
    Compute evaluation metrics.
    
    Args:
        predictions: Predicted labels
        labels: True labels
        average: Averaging strategy for multi-class
        
    Returns:
        Dictionary of metrics
    """
    return {
        'accuracy': accuracy_score(labels, predictions),
        'f1': f1_score(labels, predictions, average=average, zero_division=0),
        'precision': precision_score(labels, predictions, average=average, zero_division=0),
        'recall': recall_score(labels, predictions, average=average, zero_division=0),
    }


class ProgressTracker:
    """
    Track and display training progress.
    """
    
    def __init__(
        self,
        total_steps: int,
        log_interval: int = 100,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize progress tracker.
        
        Args:
            total_steps: Total training steps
            log_interval: Steps between log messages
            logger: Optional logger instance
        """
        self.total_steps = total_steps
        self.log_interval = log_interval
        self.logger = logger or logging.getLogger(__name__)
        
        self.start_time = time.time()
        self.step = 0
        self.epoch = 0
        self.metrics_history: List[Dict] = []
    
    def update(
        self,
        step: int,
        epoch: int,
        loss: float,
        metrics: Optional[Dict] = None
    ):
        """
        Update progress.
        
        Args:
            step: Current step
            epoch: Current epoch
            loss: Current loss
            metrics: Optional additional metrics
        """
        self.step = step
        self.epoch = epoch
        
        record = {
            'step': step,
            'epoch': epoch,
            'loss': loss,
            'timestamp': time.time() - self.start_time
        }
        if metrics:
            record.update(metrics)
        
        self.metrics_history.append(record)
        
        # CURSOR: Log at intervals
        if step % self.log_interval == 0:
            self._log_progress(loss, metrics)
    
    def _log_progress(self, loss: float, metrics: Optional[Dict]):
        """Log current progress."""
        elapsed = time.time() - self.start_time
        steps_per_sec = self.step / elapsed if elapsed > 0 else 0
        eta_seconds = (self.total_steps - self.step) / steps_per_sec if steps_per_sec > 0 else 0
        
        msg = f"Step {self.step}/{self.total_steps} | Epoch {self.epoch} | Loss: {loss:.4f}"
        
        if metrics:
            for k, v in metrics.items():
                if isinstance(v, float):
                    msg += f" | {k}: {v:.4f}"
        
        msg += f" | ETA: {self._format_time(eta_seconds)}"
        
        self.logger.info(msg)
    
    def _format_time(self, seconds: float) -> str:
        """Format seconds to human readable."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours}h {minutes}m"
        elif minutes > 0:
            return f"{minutes}m {secs}s"
        else:
            return f"{secs}s"
    
    def get_history(self) -> List[Dict]:
        """Get metrics history."""
        return self.metrics_history


def set_seed(seed: int):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_training_config(config: Dict, save_path: str):
    """Save training configuration to JSON."""
    with open(save_path, 'w') as f:
        json.dump(config, f, indent=2)


def load_training_config(load_path: str) -> Dict:
    """Load training configuration from JSON."""
    with open(load_path, 'r') as f:
        return json.load(f)


# CURSOR: Export public API
__all__ = [
    'setup_logging',
    'get_device',
    'count_parameters',
    'CheckpointManager',
    'compute_metrics',
    'ProgressTracker',
    'set_seed',
    'save_training_config',
    'load_training_config'
]



