"""
Model Optimization: Knowledge Distillation, Quantization, ONNX Export

Features:
- Knowledge distillation from ensemble to single model
- INT8/FP16 quantization
- ONNX export for optimized inference
- Target: <100ms CPU inference
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Any
from pathlib import Path


class KnowledgeDistillation:
    """
    Knowledge distillation from teacher (ensemble) to student (single model).
    
    Transfers knowledge using soft targets and intermediate representations.
    """
    
    def __init__(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        temperature: float = 3.0,
        alpha: float = 0.7,
        device: str = 'cuda'
    ):
        """
        Initialize knowledge distillation.
        
        Args:
            teacher_model: Large teacher model (ensemble)
            student_model: Smaller student model
            temperature: Softmax temperature for soft targets
            alpha: Weight for distillation loss vs hard loss
            device: Computation device
        """
        self.teacher = teacher_model.to(device)
        self.student = student_model.to(device)
        self.temperature = temperature
        self.alpha = alpha
        self.device = device
        
        # CURSOR: Freeze teacher
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
    
    def distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined distillation and hard label loss.
        
        Args:
            student_logits: Student model predictions
            teacher_logits: Teacher model predictions
            labels: Ground truth labels
            
        Returns:
            Combined loss and component losses
        """
        # CURSOR: Soft targets from teacher
        soft_targets = F.softmax(teacher_logits / self.temperature, dim=-1)
        soft_student = F.log_softmax(student_logits / self.temperature, dim=-1)
        
        # KL divergence for soft targets
        distill_loss = F.kl_div(
            soft_student,
            soft_targets,
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        # CURSOR: Hard label loss
        hard_loss = F.cross_entropy(student_logits, labels)
        
        # CURSOR: Combined loss
        total_loss = self.alpha * distill_loss + (1 - self.alpha) * hard_loss
        
        return total_loss, {
            'distill_loss': distill_loss.item(),
            'hard_loss': hard_loss.item(),
            'total_loss': total_loss.item()
        }
    
    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        optimizer: torch.optim.Optimizer
    ) -> Dict[str, float]:
        """
        Single training step for distillation.
        
        Args:
            batch: Input batch
            optimizer: Student optimizer
            
        Returns:
            Loss metrics
        """
        self.student.train()
        
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        labels = batch['labels'].to(self.device)
        
        # CURSOR: Get teacher predictions (no grad)
        with torch.no_grad():
            teacher_outputs = self.teacher(input_ids, attention_mask)
            teacher_logits = teacher_outputs['logits']
        
        # CURSOR: Get student predictions
        student_outputs = self.student(input_ids, attention_mask)
        student_logits = student_outputs['logits']
        
        # CURSOR: Compute distillation loss
        loss, metrics = self.distillation_loss(
            student_logits, teacher_logits, labels
        )
        
        # CURSOR: Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return metrics
    
    def distill(
        self,
        train_dataloader,
        val_dataloader,
        epochs: int = 10,
        learning_rate: float = 1e-4,
        save_path: Optional[str] = None
    ) -> nn.Module:
        """
        Full distillation training loop.
        
        Args:
            train_dataloader: Training data
            val_dataloader: Validation data
            epochs: Number of epochs
            learning_rate: Learning rate
            save_path: Path to save best student
            
        Returns:
            Trained student model
        """
        optimizer = torch.optim.AdamW(
            self.student.parameters(),
            lr=learning_rate
        )
        
        best_val_acc = 0.0
        
        for epoch in range(epochs):
            # Train
            self.student.train()
            train_loss = 0.0
            
            for batch in train_dataloader:
                metrics = self.train_step(batch, optimizer)
                train_loss += metrics['total_loss']
            
            train_loss /= len(train_dataloader)
            
            # Validate
            val_acc = self._evaluate(val_dataloader)
            
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f}")
            
            # Save best
            if val_acc > best_val_acc and save_path:
                best_val_acc = val_acc
                torch.save(self.student.state_dict(), save_path)
        
        return self.student
    
    def _evaluate(self, dataloader) -> float:
        """Evaluate student accuracy."""
        self.student.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.student(input_ids, attention_mask)
                preds = (outputs['logits'] > 0).long()
                
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        return correct / total if total > 0 else 0.0


class ModelQuantizer:
    """
    Quantize models for faster inference.
    
    Supports INT8 dynamic and static quantization.
    """
    
    @staticmethod
    def dynamic_quantize(
        model: nn.Module,
        dtype: torch.dtype = torch.qint8
    ) -> nn.Module:
        """
        Apply dynamic quantization.
        
        Args:
            model: Model to quantize
            dtype: Quantization dtype
            
        Returns:
            Quantized model
        """
        # CURSOR: Dynamic quantization for linear layers
        quantized = torch.quantization.quantize_dynamic(
            model.cpu(),
            {nn.Linear},
            dtype=dtype
        )
        return quantized
    
    @staticmethod
    def prepare_static_quantization(
        model: nn.Module,
        calibration_data
    ) -> nn.Module:
        """
        Prepare model for static quantization with calibration.
        
        Args:
            model: Model to quantize
            calibration_data: Representative data for calibration
            
        Returns:
            Prepared model ready for conversion
        """
        model.cpu()
        model.eval()
        
        # CURSOR: Fuse modules where possible
        # Note: Actual fusion depends on model architecture
        
        # Set quantization config
        model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        
        # Prepare for calibration
        prepared = torch.quantization.prepare(model)
        
        # CURSOR: Run calibration data through model
        with torch.no_grad():
            for batch in calibration_data:
                if isinstance(batch, dict):
                    prepared(batch['input_ids'], batch['attention_mask'])
                else:
                    prepared(batch[0], batch[1])
        
        # Convert to quantized
        quantized = torch.quantization.convert(prepared)
        
        return quantized
    
    @staticmethod
    def get_model_size(model: nn.Module) -> float:
        """Get model size in MB."""
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        return (param_size + buffer_size) / (1024 ** 2)


class ONNXExporter:
    """
    Export models to ONNX format for optimized inference.
    """
    
    @staticmethod
    def export(
        model: nn.Module,
        save_path: str,
        input_shape: Tuple[int, int] = (1, 512),
        opset_version: int = 14,
        dynamic_axes: Optional[Dict] = None
    ) -> str:
        """
        Export model to ONNX format.
        
        Args:
            model: Model to export
            save_path: Output path
            input_shape: Example input shape (batch, seq_len)
            opset_version: ONNX opset version
            dynamic_axes: Dynamic axes for variable batch size
            
        Returns:
            Path to saved ONNX model
        """
        model.cpu()
        model.eval()
        
        # CURSOR: Create dummy inputs
        dummy_input_ids = torch.zeros(input_shape, dtype=torch.long)
        dummy_attention_mask = torch.ones(input_shape, dtype=torch.long)
        
        # Default dynamic axes
        if dynamic_axes is None:
            dynamic_axes = {
                'input_ids': {0: 'batch_size', 1: 'sequence'},
                'attention_mask': {0: 'batch_size', 1: 'sequence'},
                'output': {0: 'batch_size'}
            }
        
        # CURSOR: Export
        torch.onnx.export(
            model,
            (dummy_input_ids, dummy_attention_mask),
            save_path,
            input_names=['input_ids', 'attention_mask'],
            output_names=['output'],
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            do_constant_folding=True
        )
        
        print(f"ONNX model saved to {save_path}")
        return save_path
    
    @staticmethod
    def verify_onnx(onnx_path: str) -> bool:
        """Verify ONNX model is valid."""
        try:
            import onnx
            model = onnx.load(onnx_path)
            onnx.checker.check_model(model)
            print("ONNX model is valid!")
            return True
        except Exception as e:
            print(f"ONNX verification failed: {e}")
            return False
    
    @staticmethod
    def benchmark_onnx(
        onnx_path: str,
        input_shape: Tuple[int, int] = (1, 512),
        num_runs: int = 100
    ) -> Dict[str, float]:
        """
        Benchmark ONNX model inference speed.
        
        Returns:
            Dictionary with timing statistics
        """
        try:
            import onnxruntime as ort
            import time
            
            session = ort.InferenceSession(onnx_path)
            
            # Create inputs
            input_ids = torch.zeros(input_shape, dtype=torch.long).numpy()
            attention_mask = torch.ones(input_shape, dtype=torch.long).numpy()
            
            # Warmup
            for _ in range(10):
                session.run(None, {
                    'input_ids': input_ids,
                    'attention_mask': attention_mask
                })
            
            # Benchmark
            times = []
            for _ in range(num_runs):
                start = time.perf_counter()
                session.run(None, {
                    'input_ids': input_ids,
                    'attention_mask': attention_mask
                })
                times.append((time.perf_counter() - start) * 1000)  # ms
            
            return {
                'mean_ms': sum(times) / len(times),
                'min_ms': min(times),
                'max_ms': max(times),
                'std_ms': (sum((t - sum(times)/len(times))**2 for t in times) / len(times)) ** 0.5
            }
        
        except ImportError:
            print("onnxruntime not installed")
            return {}


def optimize_for_inference(
    model: nn.Module,
    output_dir: str,
    quantize: bool = True,
    export_onnx: bool = True,
    calibration_data=None
) -> Dict[str, Any]:
    """
    Full optimization pipeline.
    
    Args:
        model: Model to optimize
        output_dir: Output directory
        quantize: Whether to quantize
        export_onnx: Whether to export ONNX
        calibration_data: Data for static quantization
        
    Returns:
        Dictionary with optimization results
    """
    os.makedirs(output_dir, exist_ok=True)
    results = {}
    
    # Original size
    original_size = ModelQuantizer.get_model_size(model)
    results['original_size_mb'] = original_size
    print(f"Original model size: {original_size:.2f} MB")
    
    # CURSOR: Quantization
    if quantize:
        print("\nApplying dynamic quantization...")
        quantized = ModelQuantizer.dynamic_quantize(model)
        quantized_size = ModelQuantizer.get_model_size(quantized)
        
        torch.save(quantized.state_dict(), f"{output_dir}/model_quantized.pt")
        results['quantized_size_mb'] = quantized_size
        results['compression_ratio'] = original_size / quantized_size
        print(f"Quantized size: {quantized_size:.2f} MB ({results['compression_ratio']:.1f}x smaller)")
    
    # CURSOR: ONNX export
    if export_onnx:
        print("\nExporting to ONNX...")
        onnx_path = f"{output_dir}/model.onnx"
        ONNXExporter.export(model, onnx_path)
        
        if ONNXExporter.verify_onnx(onnx_path):
            print("Benchmarking ONNX inference...")
            benchmark = ONNXExporter.benchmark_onnx(onnx_path)
            results['onnx_benchmark'] = benchmark
            if benchmark:
                print(f"ONNX inference: {benchmark['mean_ms']:.2f} ms (target: <100ms)")
    
    return results


# CURSOR: Export public API
__all__ = [
    'KnowledgeDistillation',
    'ModelQuantizer',
    'ONNXExporter',
    'optimize_for_inference'
]



