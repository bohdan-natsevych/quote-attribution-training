"""
Ensemble Model for Quote Attribution

3-model ensemble combining:
1. DeBERTa-v3-large (primary)
2. RoBERTa-large (robustness)
3. ELECTRA-large (discrimination)

Features:
- Weighted voting
- Confidence-weighted averaging
- Uncertainty estimation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class EnsembleConfig:
    """Configuration for ensemble model."""
    models: List[str]
    weights: Optional[List[float]] = None
    voting_strategy: str = 'weighted_average'  # 'voting', 'weighted_average', 'max_confidence'
    temperature: float = 1.0


class EnsembleSpeakerModel(nn.Module):
    """
    Ensemble of multiple speaker attribution models.
    
    Combines predictions from multiple models for improved robustness.
    """
    
    DEFAULT_MODELS = [
        'microsoft/deberta-v3-large',
        'roberta-large',
        'google/electra-large-discriminator'
    ]
    
    def __init__(
        self,
        model_configs: Optional[List[Dict]] = None,
        weights: Optional[List[float]] = None,
        voting_strategy: str = 'weighted_average',
        hidden_size: int = 1024,
        dropout: float = 0.2
    ):
        """
        Initialize ensemble model.
        
        Args:
            model_configs: List of model configurations
            weights: Weights for each model (default: equal)
            voting_strategy: How to combine predictions
            hidden_size: Hidden size for classifiers
            dropout: Dropout rate
        """
        super().__init__()
        
        self.voting_strategy = voting_strategy
        
        # CURSOR: Create individual models
        self.models = nn.ModuleList()
        self.model_names = []
        
        if model_configs is None:
            model_configs = [{'name': m} for m in self.DEFAULT_MODELS]
        
        for config in model_configs:
            model = self._create_single_model(
                config.get('name', 'microsoft/deberta-v3-large'),
                hidden_size,
                dropout
            )
            self.models.append(model)
            self.model_names.append(config.get('name', 'unknown'))
        
        # CURSOR: Model weights
        if weights is None:
            weights = [1.0 / len(self.models)] * len(self.models)
        self.register_buffer('weights', torch.tensor(weights))
        
        # CURSOR: Learnable combination layer
        self.combiner = nn.Sequential(
            nn.Linear(len(self.models), len(self.models) * 2),
            nn.ReLU(),
            nn.Linear(len(self.models) * 2, len(self.models)),
            nn.Softmax(dim=-1)
        )
    
    def _create_single_model(
        self,
        model_name: str,
        hidden_size: int,
        dropout: float
    ) -> nn.Module:
        """Create a single model component."""
        from transformers import AutoModel, AutoTokenizer
        
        class SingleModel(nn.Module):
            def __init__(self, name, h_size, drop):
                super().__init__()
                self.encoder = AutoModel.from_pretrained(name)
                encoder_hidden = self.encoder.config.hidden_size
                
                self.classifier = nn.Sequential(
                    nn.Linear(encoder_hidden, h_size),
                    nn.LayerNorm(h_size),
                    nn.GELU(),
                    nn.Dropout(drop),
                    nn.Linear(h_size, 512),
                    nn.LayerNorm(512),
                    nn.GELU(),
                    nn.Dropout(drop),
                    nn.Linear(512, 1)
                )
            
            def forward(self, input_ids, attention_mask):
                outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
                cls_output = outputs.last_hidden_state[:, 0, :]
                return self.classifier(cls_output)
        
        return SingleModel(model_name, hidden_size, dropout)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_individual: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass combining all models.
        
        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            return_individual: Whether to return individual model outputs
            
        Returns:
            Dictionary with combined logits and optional individual outputs
        """
        individual_logits = []
        
        # CURSOR: Get predictions from each model
        for model in self.models:
            logits = model(input_ids, attention_mask)
            individual_logits.append(logits)
        
        # Stack: [batch, num_models]
        stacked = torch.cat(individual_logits, dim=-1)
        
        # CURSOR: Combine based on strategy
        if self.voting_strategy == 'voting':
            # Hard voting
            votes = (stacked > 0).float()
            combined = votes.sum(dim=-1, keepdim=True) / len(self.models)
        
        elif self.voting_strategy == 'weighted_average':
            # Weighted average
            combined = (stacked * self.weights.unsqueeze(0)).sum(dim=-1, keepdim=True)
        
        elif self.voting_strategy == 'max_confidence':
            # Take prediction with highest confidence
            confidences = torch.abs(stacked)
            max_idx = confidences.argmax(dim=-1, keepdim=True)
            combined = stacked.gather(1, max_idx)
        
        elif self.voting_strategy == 'learned':
            # Learn optimal combination
            learned_weights = self.combiner(stacked)
            combined = (stacked * learned_weights).sum(dim=-1, keepdim=True)
        
        else:
            # Default to simple average
            combined = stacked.mean(dim=-1, keepdim=True)
        
        result = {'logits': combined.squeeze(-1)}
        
        if return_individual:
            result['individual_logits'] = stacked
            result['model_names'] = self.model_names
        
        return result
    
    def predict_with_uncertainty(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict with uncertainty estimation.
        
        Returns:
            predictions: Combined predictions
            mean_confidence: Mean confidence across models
            std_confidence: Standard deviation (uncertainty)
        """
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask, return_individual=True)
            
            individual = outputs['individual_logits']
            probs = torch.sigmoid(individual)
            
            mean_probs = probs.mean(dim=-1)
            std_probs = probs.std(dim=-1)
            
            predictions = (mean_probs > 0.5).long()
        
        return predictions, mean_probs, std_probs
    
    def get_model_count(self) -> int:
        """Return number of models in ensemble."""
        return len(self.models)
    
    def freeze_model(self, index: int):
        """Freeze a specific model's parameters."""
        for param in self.models[index].parameters():
            param.requires_grad = False
    
    def unfreeze_model(self, index: int):
        """Unfreeze a specific model's parameters."""
        for param in self.models[index].parameters():
            param.requires_grad = True


def create_ensemble(
    model_names: Optional[List[str]] = None,
    weights: Optional[List[float]] = None,
    device: str = 'cuda'
) -> EnsembleSpeakerModel:
    """
    Factory function to create ensemble model.
    
    Args:
        model_names: List of model names
        weights: Model weights
        device: Device to place model on
        
    Returns:
        Initialized ensemble model
    """
    if model_names is None:
        model_names = EnsembleSpeakerModel.DEFAULT_MODELS
    
    model_configs = [{'name': name} for name in model_names]
    
    ensemble = EnsembleSpeakerModel(
        model_configs=model_configs,
        weights=weights,
        voting_strategy='weighted_average'
    )
    
    return ensemble.to(device)


# CURSOR: Export public API
__all__ = [
    'EnsembleSpeakerModel',
    'EnsembleConfig',
    'create_ensemble'
]



