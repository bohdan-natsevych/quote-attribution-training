"""
Advanced Loss Functions for Quote Attribution

Includes:
- Focal Loss: Better handling of class imbalance
- Label Smoothing: Regularization to prevent overconfidence
- R-Drop: Consistency regularization for better generalization
- Combined loss with all techniques
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    
    Focal loss down-weights easy examples and focuses on hard negatives.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Reference: Lin et al. "Focal Loss for Dense Object Detection" (2017)
    """
    
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        reduction: str = 'mean',
        ignore_index: int = -100
    ):
        """
        Initialize Focal Loss.
        
        Args:
            gamma: Focusing parameter (higher = more focus on hard examples)
            alpha: Class weights tensor, shape [num_classes]
            reduction: 'none', 'mean', or 'sum'
        """
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.ignore_index = ignore_index
    
    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            inputs: Predictions [batch, num_classes] or [batch]
            targets: Ground truth labels [batch]
            
        Returns:
            Focal loss value
        """
        # CURSOR: Handle both multi-class and binary cases
        if inputs.dim() == 1:
            # Binary case - convert to 2-class
            inputs = torch.stack([1 - inputs, inputs], dim=-1)

        valid = targets != self.ignore_index
        if valid.sum().item() == 0:
            return inputs.sum() * 0.0

        v_inputs = inputs[valid]
        v_targets = targets[valid]

        # Compute standard cross-entropy loss (no reduction)
        ce_loss = F.cross_entropy(v_inputs, v_targets, reduction='none')

        # Get probabilities for focal weighting
        p = F.softmax(v_inputs, dim=-1)
        p_t = p.gather(1, v_targets.unsqueeze(-1)).squeeze(-1)

        # CURSOR: Compute focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma
        focal_loss = focal_weight * ce_loss

        # CURSOR: Apply class weights if provided
        if self.alpha is not None:
            alpha_t = self.alpha.to(v_inputs.device).gather(0, v_targets)
            focal_loss = alpha_t * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        if self.reduction == 'sum':
            return focal_loss.sum()

        # reduction='none' -> return a vector aligned with batch size
        out = torch.zeros(targets.shape[0], device=inputs.device, dtype=focal_loss.dtype)
        out[valid] = focal_loss
        return out


class LabelSmoothingLoss(nn.Module):
    """
    Cross-entropy loss with label smoothing.
    
    Smooths the target distribution to prevent overconfidence.
    """
    
    def __init__(
        self,
        smoothing: float = 0.1,
        reduction: str = 'mean',
        ignore_index: int = -100
    ):
        """
        Initialize Label Smoothing Loss.
        
        Args:
            smoothing: Label smoothing factor (0 = no smoothing, 1 = uniform)
            reduction: 'none', 'mean', or 'sum'
        """
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction
        self.ignore_index = ignore_index
    
    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute label smoothing loss.
        
        Args:
            inputs: Predictions [batch, num_classes]
            targets: Ground truth labels [batch]
            
        Returns:
            Label smoothing loss value
        """
        valid = targets != self.ignore_index
        if valid.sum().item() == 0:
            return inputs.sum() * 0.0

        v_inputs = inputs[valid]
        v_targets = targets[valid]

        # CURSOR: Efficient label-smoothed NLL without constructing dense one-hot targets.
        log_probs = F.log_softmax(v_inputs, dim=-1)
        nll_loss = -log_probs.gather(dim=-1, index=v_targets.unsqueeze(-1)).squeeze(-1)
        smooth_loss = -log_probs.mean(dim=-1)
        loss = (1.0 - self.smoothing) * nll_loss + self.smoothing * smooth_loss

        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()

        out = torch.zeros(targets.shape[0], device=inputs.device, dtype=loss.dtype)
        out[valid] = loss
        return out


class RDropLoss(nn.Module):
    """
    R-Drop: Regularized Dropout for Neural Networks.
    
    Performs two forward passes with different dropout masks and
    minimizes KL divergence between the two output distributions.
    
    Reference: Wu et al. "R-Drop: Regularized Dropout for Neural Networks" (2021)
    """
    
    def __init__(self, alpha: float = 0.7):
        """
        Initialize R-Drop Loss.
        
        Args:
            alpha: Weight for KL divergence term (higher = more regularization)
        """
        super().__init__()
        self.alpha = alpha
    
    def compute_kl_divergence(
        self,
        p_logits: torch.Tensor,
        q_logits: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute symmetric KL divergence between two distributions.
        
        Args:
            p_logits: First distribution logits [batch, num_classes]
            q_logits: Second distribution logits [batch, num_classes]
            
        Returns:
            Symmetric KL divergence
        """
        p = F.softmax(p_logits, dim=-1)
        q = F.softmax(q_logits, dim=-1)
        
        p_log = F.log_softmax(p_logits, dim=-1)
        q_log = F.log_softmax(q_logits, dim=-1)
        
        # CURSOR: Symmetric KL: KL(p||q) + KL(q||p)
        kl_pq = F.kl_div(q_log, p, reduction='batchmean')
        kl_qp = F.kl_div(p_log, q, reduction='batchmean')
        
        return (kl_pq + kl_qp) / 2
    
    def forward(
        self,
        logits1: torch.Tensor,
        logits2: torch.Tensor,
        targets: torch.Tensor,
        criterion: nn.Module
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute R-Drop loss.
        
        Args:
            logits1: First forward pass output [batch, num_classes]
            logits2: Second forward pass output [batch, num_classes]
            targets: Ground truth labels [batch]
            criterion: Base loss function (e.g., CrossEntropy, FocalLoss)
            
        Returns:
            total_loss: Combined loss
            ce_loss: Average cross-entropy loss
            kl_loss: KL divergence loss
        """
        # CURSOR: Compute cross-entropy for both passes
        ce_loss1 = criterion(logits1, targets)
        ce_loss2 = criterion(logits2, targets)
        ce_loss = (ce_loss1 + ce_loss2) / 2
        
        # CURSOR: Compute KL divergence
        kl_loss = self.compute_kl_divergence(logits1, logits2)
        
        # CURSOR: Combined loss
        total_loss = ce_loss + self.alpha * kl_loss
        
        return total_loss, ce_loss, kl_loss


class CombinedLoss(nn.Module):
    """
    Combined loss with Focal Loss, Label Smoothing, and R-Drop.
    
    This is the recommended loss for maximum performance training.
    """
    
    def __init__(
        self,
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.1,
        r_drop_alpha: float = 0.7,
        class_weights: Optional[torch.Tensor] = None,
        use_focal: bool = True,
        use_label_smoothing: bool = True,
        use_r_drop: bool = True,
        ignore_index: int = -100
    ):
        """
        Initialize Combined Loss.
        
        Args:
            focal_gamma: Focal loss gamma parameter
            label_smoothing: Label smoothing factor
            r_drop_alpha: R-Drop regularization weight
            class_weights: Optional class weights for imbalanced data
            use_focal: Whether to use focal loss
            use_label_smoothing: Whether to use label smoothing
            use_r_drop: Whether to use R-Drop
        """
        super().__init__()
        
        self.use_focal = use_focal
        self.use_label_smoothing = use_label_smoothing
        self.use_r_drop = use_r_drop
        self.ignore_index = ignore_index
        self.focal_gamma = focal_gamma
        self.label_smoothing = label_smoothing
        self.class_weights = class_weights
        self.r_drop_alpha = r_drop_alpha
        
        # CURSOR: Keep modules for backwards compatibility / external callers.
        self.focal_loss = FocalLoss(
            gamma=focal_gamma,
            alpha=class_weights,
            reduction='mean',
            ignore_index=ignore_index
        )
        self.label_smoothing_loss = LabelSmoothingLoss(
            smoothing=label_smoothing,
            reduction='mean',
            ignore_index=ignore_index
        )
        self.r_drop = RDropLoss(alpha=r_drop_alpha)
        self.ce_loss = nn.CrossEntropyLoss(
            weight=class_weights,
            reduction='mean',
            ignore_index=ignore_index
        )
    
    def get_base_criterion(self) -> nn.Module:
        """Get the base loss criterion for external callers."""
        if self.use_label_smoothing and self.label_smoothing > 0:
            return self.label_smoothing_loss
        if self.use_focal and self.focal_gamma > 0:
            return self.focal_loss
        return self.ce_loss

    def _per_example_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """CURSOR: Compute per-example loss with optional label smoothing + focal weighting."""
        log_probs = F.log_softmax(logits, dim=-1)
        nll = -log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

        if self.use_label_smoothing and self.label_smoothing > 0:
            smooth = -log_probs.mean(dim=-1)
            loss = (1.0 - self.label_smoothing) * nll + self.label_smoothing * smooth
        else:
            loss = nll

        if self.use_focal and self.focal_gamma > 0:
            p_t = log_probs.exp().gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
            loss = ((1.0 - p_t) ** self.focal_gamma) * loss

        if self.class_weights is not None:
            alpha_t = self.class_weights.to(logits.device).gather(0, targets)
            loss = alpha_t * loss

        return loss
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        logits2: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute combined loss.
        
        Args:
            logits: First forward pass output [batch, num_classes]
            targets: Ground truth labels [batch]
            logits2: Optional second forward pass output for R-Drop
            
        Returns:
            Combined loss value
        """
        valid = targets != self.ignore_index
        if valid.sum().item() == 0:
            return logits.sum() * 0.0

        v_logits = logits[valid]
        v_targets = targets[valid]

        # CURSOR: Optional R-Drop when logits2 is provided and alpha > 0.
        if self.use_r_drop and logits2 is not None and self.r_drop_alpha > 0:
            v_logits2 = logits2[valid]
            loss1 = self._per_example_loss(v_logits, v_targets).mean()
            loss2 = self._per_example_loss(v_logits2, v_targets).mean()
            ce_loss = (loss1 + loss2) / 2.0
            kl_loss = self.r_drop.compute_kl_divergence(v_logits, v_logits2)
            return ce_loss + self.r_drop_alpha * kl_loss

        return self._per_example_loss(v_logits, v_targets).mean()
    
    def forward_with_r_drop(
        self,
        model: nn.Module,
        batch: dict,
        device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with R-Drop (two forward passes).
        
        Args:
            model: The speaker attribution model
            batch: Input batch dictionary
            device: Computation device
            
        Returns:
            loss: Combined loss
            logits: Average logits for metrics
        """
        # CURSOR: Move batch to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        quote_mask = batch['quote_mask'].to(device)
        candidate_masks = batch['candidate_masks'].to(device)
        targets = batch['labels'].to(device)
        
        candidate_attention_mask = batch.get('candidate_attention_mask')
        if candidate_attention_mask is not None:
            candidate_attention_mask = candidate_attention_mask.to(device)
        
        # CURSOR: First forward pass
        model.train()
        logits1, _ = model(
            input_ids, attention_mask, quote_mask,
            candidate_masks, candidate_attention_mask
        )
        
        # CURSOR: Second forward pass (different dropout mask)
        logits2, _ = model(
            input_ids, attention_mask, quote_mask,
            candidate_masks, candidate_attention_mask
        )
        
        # CURSOR: Compute combined loss
        loss = self.forward(logits1, targets, logits2)
        
        # Return average logits for metrics
        avg_logits = (logits1 + logits2) / 2
        
        return loss, avg_logits


def compute_class_weights(
    labels: torch.Tensor,
    num_classes: Optional[int] = None
) -> torch.Tensor:
    """
    Compute balanced class weights from label distribution.
    
    Args:
        labels: Tensor of labels
        num_classes: Number of classes (inferred if not provided)
        
    Returns:
        Class weights tensor
    """
    if num_classes is None:
        num_classes = int(labels.max().item()) + 1
    
    # Count occurrences of each class
    counts = torch.bincount(labels, minlength=num_classes).float()
    
    # CURSOR: Compute balanced weights: n_samples / (n_classes * n_samples_per_class)
    total = counts.sum()
    weights = total / (num_classes * counts.clamp(min=1))
    
    # Normalize so mean weight is 1
    weights = weights / weights.mean()
    
    return weights


# CURSOR: Export public API
__all__ = [
    'FocalLoss',
    'LabelSmoothingLoss',
    'RDropLoss',
    'CombinedLoss',
    'compute_class_weights'
]



