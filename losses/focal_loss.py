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
        reduction: str = 'mean'
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
        
        # Compute standard cross-entropy loss (no reduction)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Get probabilities
        p = F.softmax(inputs, dim=-1)
        p_t = p.gather(1, targets.unsqueeze(-1)).squeeze(-1)
        
        # CURSOR: Compute focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma
        
        # Apply focal weight to cross-entropy loss
        focal_loss = focal_weight * ce_loss
        
        # CURSOR: Apply class weights if provided
        if self.alpha is not None:
            alpha_t = self.alpha.to(inputs.device).gather(0, targets)
            focal_loss = alpha_t * focal_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingLoss(nn.Module):
    """
    Cross-entropy loss with label smoothing.
    
    Smooths the target distribution to prevent overconfidence.
    """
    
    def __init__(
        self,
        smoothing: float = 0.1,
        reduction: str = 'mean'
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
        num_classes = inputs.size(-1)
        
        # CURSOR: Create smoothed target distribution
        # One-hot encode targets
        smooth_targets = torch.zeros_like(inputs)
        smooth_targets.fill_(self.smoothing / (num_classes - 1))
        smooth_targets.scatter_(1, targets.unsqueeze(-1), 1 - self.smoothing)
        
        # Compute log probabilities
        log_probs = F.log_softmax(inputs, dim=-1)
        
        # CURSOR: Compute KL divergence (which equals cross-entropy for one-hot)
        loss = -(smooth_targets * log_probs).sum(dim=-1)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


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
        use_r_drop: bool = True
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
        
        # CURSOR: Initialize loss components
        if use_focal:
            self.focal_loss = FocalLoss(
                gamma=focal_gamma,
                alpha=class_weights,
                reduction='mean'
            )
        
        if use_label_smoothing:
            self.label_smoothing_loss = LabelSmoothingLoss(
                smoothing=label_smoothing,
                reduction='mean'
            )
        
        if use_r_drop:
            self.r_drop = RDropLoss(alpha=r_drop_alpha)
        
        # Fallback to standard cross-entropy
        self.ce_loss = nn.CrossEntropyLoss(
            weight=class_weights,
            reduction='mean'
        )
    
    def get_base_criterion(self) -> nn.Module:
        """Get the base loss criterion (focal or label smoothing)."""
        if self.use_focal:
            return self.focal_loss
        elif self.use_label_smoothing:
            return self.label_smoothing_loss
        else:
            return self.ce_loss
    
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
        criterion = self.get_base_criterion()
        
        # CURSOR: If R-Drop is enabled and we have two forward passes
        if self.use_r_drop and logits2 is not None:
            total_loss, ce_loss, kl_loss = self.r_drop(
                logits, logits2, targets, criterion
            )
            return total_loss
        
        # CURSOR: Single forward pass - use base criterion
        return criterion(logits, targets)
    
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



