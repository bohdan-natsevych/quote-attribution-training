"""
Confidence Calibration for Quote Attribution

Implements temperature scaling and calibrated prediction:
- TemperatureScaling: Learn optimal temperature on validation set
- CalibratedPredictor: Make predictions with calibrated confidence scores
- Reliability diagrams for calibration visualization

Well-calibrated models provide confidence scores that reflect true accuracy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class CalibrationMetrics:
    """Container for calibration metrics."""
    ece: float  # Expected Calibration Error
    mce: float  # Maximum Calibration Error
    reliability_diagram: Dict[str, List[float]]
    temperature: float


class TemperatureScaling(nn.Module):
    """
    Temperature scaling for model calibration.
    
    Learns a single scalar temperature parameter that scales logits
    to produce calibrated probability estimates.
    
    Reference: Guo et al. "On Calibration of Modern Neural Networks" (2017)
    """
    
    def __init__(self, initial_temperature: float = 1.5):
        """
        Initialize temperature scaling.
        
        Args:
            initial_temperature: Initial temperature value
        """
        super().__init__()
        # CURSOR: Temperature parameter (must be > 0)
        self.temperature = nn.Parameter(torch.ones(1) * initial_temperature)
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply temperature scaling to logits.
        
        Args:
            logits: Raw model logits [batch, num_classes]
            
        Returns:
            Scaled logits
        """
        return logits / self.temperature.clamp(min=0.01)
    
    def calibrate(
        self,
        val_logits: torch.Tensor,
        val_labels: torch.Tensor,
        lr: float = 0.01,
        max_iter: int = 100
    ) -> float:
        """
        Learn optimal temperature on validation set.
        
        Uses LBFGS optimizer to minimize NLL on validation data.
        
        Args:
            val_logits: Validation logits [num_samples, num_classes]
            val_labels: Validation labels [num_samples]
            lr: Learning rate for LBFGS
            max_iter: Maximum iterations
            
        Returns:
            Final temperature value
        """
        # CURSOR: Reset temperature
        self.temperature.data.fill_(1.5)
        
        # CURSOR: Use LBFGS optimizer (good for small param problems)
        optimizer = optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)
        
        nll_criterion = nn.CrossEntropyLoss()
        
        def eval_step():
            optimizer.zero_grad()
            scaled_logits = self.forward(val_logits)
            loss = nll_criterion(scaled_logits, val_labels)
            loss.backward()
            return loss
        
        optimizer.step(eval_step)
        
        return self.temperature.item()
    
    def get_temperature(self) -> float:
        """Return current temperature value."""
        return self.temperature.item()


class CalibratedPredictor:
    """
    Wrapper for calibrated predictions.
    
    Provides predictions with reliable confidence scores.
    """
    
    # CURSOR: Confidence thresholds
    HIGH_CONFIDENCE_THRESHOLD = 0.8
    MEDIUM_CONFIDENCE_THRESHOLD = 0.6
    LOW_CONFIDENCE_THRESHOLD = 0.4
    
    def __init__(
        self,
        model: nn.Module,
        temperature_scaler: Optional[TemperatureScaling] = None,
        device: str = 'cpu'
    ):
        """
        Initialize calibrated predictor.
        
        Args:
            model: The speaker attribution model
            temperature_scaler: Calibrated temperature scaler
            device: Computation device
        """
        self.model = model
        self.temperature_scaler = temperature_scaler or TemperatureScaling()
        self.device = device
        
        self.model.to(device)
        self.model.eval()
    
    def calibrate_on_validation(
        self,
        val_dataloader,
        progress_callback=None
    ) -> float:
        """
        Calibrate temperature on validation data.
        
        Args:
            val_dataloader: Validation DataLoader
            progress_callback: Optional callback for progress updates
            
        Returns:
            Calibrated temperature value
        """
        all_logits = []
        all_labels = []
        
        self.model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_dataloader):
                # CURSOR: Get model predictions
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                quote_mask = batch['quote_mask'].to(self.device)
                candidate_masks = batch['candidate_masks'].to(self.device)
                labels = batch['labels']
                
                candidate_attention_mask = batch.get('candidate_attention_mask')
                if candidate_attention_mask is not None:
                    candidate_attention_mask = candidate_attention_mask.to(self.device)
                
                logits, _ = self.model(
                    input_ids, attention_mask, quote_mask,
                    candidate_masks, candidate_attention_mask
                )
                
                all_logits.append(logits.cpu())
                all_labels.append(labels)
                
                if progress_callback:
                    progress_callback(batch_idx)
        
        # CURSOR: Concatenate all predictions
        val_logits = torch.cat(all_logits, dim=0)
        val_labels = torch.cat(all_labels, dim=0)
        
        # CURSOR: Calibrate temperature
        temperature = self.temperature_scaler.calibrate(val_logits, val_labels)
        
        return temperature
    
    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        quote_mask: torch.Tensor,
        candidate_masks: torch.Tensor,
        candidate_attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        """
        Make calibrated predictions.
        
        Returns:
            predictions: Predicted speaker indices [batch]
            probabilities: Calibrated probabilities [batch, num_candidates]
            confidence_levels: List of confidence level strings
        """
        self.model.eval()
        
        with torch.no_grad():
            # CURSOR: Get raw logits
            logits, _ = self.model(
                input_ids.to(self.device),
                attention_mask.to(self.device),
                quote_mask.to(self.device),
                candidate_masks.to(self.device),
                candidate_attention_mask.to(self.device) if candidate_attention_mask is not None else None
            )
            
            # CURSOR: Apply temperature scaling
            scaled_logits = self.temperature_scaler(logits)
            
            # CURSOR: Get calibrated probabilities
            probabilities = F.softmax(scaled_logits, dim=-1)
            
            # CURSOR: Get predictions
            predictions = torch.argmax(probabilities, dim=-1)
            max_probs = probabilities.max(dim=-1)[0]
            
            # CURSOR: Determine confidence levels
            confidence_levels = []
            for p in max_probs.cpu().numpy():
                if p >= self.HIGH_CONFIDENCE_THRESHOLD:
                    confidence_levels.append("HIGH")
                elif p >= self.MEDIUM_CONFIDENCE_THRESHOLD:
                    confidence_levels.append("MEDIUM")
                elif p >= self.LOW_CONFIDENCE_THRESHOLD:
                    confidence_levels.append("LOW")
                else:
                    confidence_levels.append("VERY_LOW")
        
        return predictions.cpu(), probabilities.cpu(), confidence_levels
    
    def predict_with_uncertainty(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        quote_mask: torch.Tensor,
        candidate_masks: torch.Tensor,
        candidate_attention_mask: Optional[torch.Tensor] = None,
        num_mc_samples: int = 10
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict with uncertainty estimation using MC Dropout.
        
        Args:
            ... (same as predict)
            num_mc_samples: Number of Monte Carlo samples
            
        Returns:
            predictions: Mean predictions [batch]
            mean_probs: Mean probabilities [batch, num_candidates]
            std_probs: Standard deviation of probabilities
        """
        # CURSOR: Enable dropout for MC sampling
        self.model.train()
        
        all_probs = []
        
        with torch.no_grad():
            for _ in range(num_mc_samples):
                logits, _ = self.model(
                    input_ids.to(self.device),
                    attention_mask.to(self.device),
                    quote_mask.to(self.device),
                    candidate_masks.to(self.device),
                    candidate_attention_mask.to(self.device) if candidate_attention_mask is not None else None
                )
                
                scaled_logits = self.temperature_scaler(logits)
                probs = F.softmax(scaled_logits, dim=-1)
                all_probs.append(probs.cpu())
        
        # CURSOR: Back to eval mode
        self.model.eval()
        
        # CURSOR: Compute statistics
        stacked = torch.stack(all_probs, dim=0)
        mean_probs = stacked.mean(dim=0)
        std_probs = stacked.std(dim=0)
        
        predictions = torch.argmax(mean_probs, dim=-1)
        
        return predictions, mean_probs, std_probs


def compute_calibration_metrics(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    num_bins: int = 10
) -> CalibrationMetrics:
    """
    Compute calibration metrics.
    
    Args:
        probabilities: Predicted probabilities [num_samples, num_classes]
        labels: True labels [num_samples]
        num_bins: Number of bins for reliability diagram
        
    Returns:
        CalibrationMetrics dataclass
    """
    # CURSOR: Get confidence (max probability) and predictions
    confidences, predictions = probabilities.max(dim=-1)
    accuracies = predictions.eq(labels).float()
    
    # CURSOR: Convert to numpy
    confidences = confidences.numpy()
    accuracies = accuracies.numpy()
    
    # CURSOR: Create bins
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    # CURSOR: Compute metrics per bin
    bin_confidences = []
    bin_accuracies = []
    bin_counts = []
    
    ece = 0.0
    mce = 0.0
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Samples in this bin
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            avg_confidence = confidences[in_bin].mean()
            avg_accuracy = accuracies[in_bin].mean()
            
            bin_confidences.append(float(avg_confidence))
            bin_accuracies.append(float(avg_accuracy))
            bin_counts.append(int(in_bin.sum()))
            
            # ECE contribution
            gap = abs(avg_accuracy - avg_confidence)
            ece += prop_in_bin * gap
            mce = max(mce, gap)
        else:
            bin_confidences.append(0.0)
            bin_accuracies.append(0.0)
            bin_counts.append(0)
    
    return CalibrationMetrics(
        ece=float(ece),
        mce=float(mce),
        reliability_diagram={
            'bin_confidences': bin_confidences,
            'bin_accuracies': bin_accuracies,
            'bin_counts': bin_counts,
            'bin_edges': list(bin_boundaries)
        },
        temperature=1.0  # Will be updated after calibration
    )


def print_calibration_report(metrics: CalibrationMetrics):
    """Print a formatted calibration report."""
    print("\n" + "=" * 50)
    print("CALIBRATION REPORT")
    print("=" * 50)
    print(f"Expected Calibration Error (ECE): {metrics.ece:.4f}")
    print(f"Maximum Calibration Error (MCE): {metrics.mce:.4f}")
    print(f"Temperature: {metrics.temperature:.4f}")
    print("\nReliability Diagram:")
    print("-" * 50)
    print(f"{'Bin':<10} {'Confidence':<12} {'Accuracy':<12} {'Count':<10}")
    print("-" * 50)
    
    diagram = metrics.reliability_diagram
    for i, (conf, acc, count) in enumerate(zip(
        diagram['bin_confidences'],
        diagram['bin_accuracies'],
        diagram['bin_counts']
    )):
        bin_range = f"{diagram['bin_edges'][i]:.1f}-{diagram['bin_edges'][i+1]:.1f}"
        print(f"{bin_range:<10} {conf:<12.3f} {acc:<12.3f} {count:<10}")
    
    print("=" * 50 + "\n")


# CURSOR: Export public API
__all__ = [
    'TemperatureScaling',
    'CalibratedPredictor',
    'CalibrationMetrics',
    'compute_calibration_metrics',
    'print_calibration_report'
]



