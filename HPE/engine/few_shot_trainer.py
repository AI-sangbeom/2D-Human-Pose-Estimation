import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple
import os
import time
from progress.bar import Bar

from models.fskd import FewShotFSKD, MetaLearningFSKD
from utils import AverageMeter, printM


class FewShotTrainer:
    """
    Trainer for few-shot keypoint detection
    Handles episodic training and evaluation
    """
    
    def __init__(
        self,
        model: FewShotFSKD,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        device: str = 'cuda',
        save_dir: str = './checkpoints',
        log_interval: int = 10,
        save_interval: int = 100
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_dir = save_dir
        self.log_interval = log_interval
        self.save_interval = save_interval
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Initialize optimizer
        if optimizer is None:
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=1e-3,
                weight_decay=1e-4
            )
        else:
            self.optimizer = optimizer
        
        # Initialize scheduler
        self.scheduler = scheduler
        
        # Training metrics
        self.train_metrics = {
            'loss': AverageMeter(),
            'keypoint_error': AverageMeter(),
            'accuracy': AverageMeter()
        }
        
        # Validation metrics
        self.val_metrics = {
            'loss': AverageMeter(),
            'keypoint_error': AverageMeter(),
            'accuracy': AverageMeter()
        }
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        printM(f"Few-shot trainer initialized with {device}")
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        printM(f"\n=== Epoch {epoch} Training ===")
        
        # Reset metrics
        for metric in self.train_metrics.values():
            metric.reset()
        
        self.model.train()
        
        n_episodes = len(self.train_loader)
        if n_episodes == 0:
            printM("No training episodes found!")
            return {k: 0.0 for k in self.train_metrics.keys()}
        
        bar = Bar('Training', max=n_episodes)
        
        for batch_idx, episode_batch in enumerate(self.train_loader):
            # Move to device
            support_images = episode_batch['support_images'].to(self.device)
            support_keypoints = episode_batch['support_keypoints'].to(self.device)
            query_images = episode_batch['query_images'].to(self.device)
            query_keypoints = episode_batch['query_keypoints'].to(self.device)
            
            # Skip empty batches
            if support_images.numel() == 0 or query_images.numel() == 0:
                continue
            
            # Forward pass
            try:
                result = self.model(
                    support_images=support_images,
                    support_keypoints=support_keypoints,
                    query_images=query_images
                )
                
                predictions = result['predictions']
                
                # Compute loss
                loss = self._compute_loss(predictions, query_keypoints)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                # Update metrics
                self.train_metrics['loss'].update(loss.item(), support_images.size(0))
                
                # Additional metrics (if available)
                if 'keypoints' in predictions:
                    keypoint_error = self._compute_keypoint_error(
                        predictions['keypoints'], query_keypoints
                    )
                    self.train_metrics['keypoint_error'].update(keypoint_error.item(), query_images.size(0))
                
                if 'confidence' in predictions:
                    accuracy = self._compute_accuracy(predictions, query_keypoints)
                    self.train_metrics['accuracy'].update(accuracy.item(), query_images.size(0))
                
                # Logging
                if batch_idx % self.log_interval == 0:
                    log_msg = f"Epoch {epoch} [{batch_idx}/{n_episodes}] "
                    log_msg += f"Loss: {self.train_metrics['loss'].avg:.4f} "
                    if self.train_metrics['keypoint_error'].count > 0:
                        log_msg += f"KP Error: {self.train_metrics['keypoint_error'].avg:.4f} "
                    if self.train_metrics['accuracy'].count > 0:
                        log_msg += f"Accuracy: {self.train_metrics['accuracy'].avg:.4f}"
                    
                    printM(log_msg)
                
                self.global_step += 1
                
            except Exception as e:
                printM(f"Error in training batch {batch_idx}: {e}")
                continue
            
            # Update progress bar
            bar.suffix = (
                f'Epoch: {epoch} | '
                f'Batch: {batch_idx+1}/{n_episodes} | '
                f'Loss: {self.train_metrics["loss"].avg:.4f} | '
                f'KP Error: {self.train_metrics["keypoint_error"].avg:.4f} | '
                f'Acc: {self.train_metrics["accuracy"].avg:.4f}'
            )
            bar.next()
        
        bar.finish()
        
        # Step scheduler
        if self.scheduler is not None:
            self.scheduler.step()
        
        self.current_epoch = epoch
        
        return {k: v.avg for k, v in self.train_metrics.items()}
    
    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        """Validate for one epoch"""
        if self.val_loader is None:
            printM("No validation loader provided!")
            return {k: 0.0 for k in self.val_metrics.keys()}
        
        printM(f"\n=== Epoch {epoch} Validation ===")
        
        # Reset metrics
        for metric in self.val_metrics.values():
            metric.reset()
        
        self.model.eval()
        
        n_episodes = len(self.val_loader)
        if n_episodes == 0:
            printM("No validation episodes found!")
            return {k: 0.0 for k in self.val_metrics.keys()}
        
        bar = Bar('Validation', max=n_episodes)
        
        with torch.no_grad():
            for batch_idx, episode_batch in enumerate(self.val_loader):
                # Move to device
                support_images = episode_batch['support_images'].to(self.device)
                support_keypoints = episode_batch['support_keypoints'].to(self.device)
                query_images = episode_batch['query_images'].to(self.device)
                query_keypoints = episode_batch['query_keypoints'].to(self.device)
                
                # Skip empty batches
                if support_images.numel() == 0 or query_images.numel() == 0:
                    continue
                
                try:
                    # Forward pass
                    result = self.model(
                        support_images=support_images,
                        support_keypoints=support_keypoints,
                        query_images=query_images
                    )
                    
                    predictions = result['predictions']
                    
                    # Compute loss
                    loss = self._compute_loss(predictions, query_keypoints)
                    
                    # Update metrics
                    self.val_metrics['loss'].update(loss.item(), support_images.size(0))
                    
                    # Additional metrics
                    if 'keypoints' in predictions:
                        keypoint_error = self._compute_keypoint_error(
                            predictions['keypoints'], query_keypoints
                        )
                        self.val_metrics['keypoint_error'].update(keypoint_error.item(), query_images.size(0))
                    
                    if 'confidence' in predictions:
                        accuracy = self._compute_accuracy(predictions, query_keypoints)
                        self.val_metrics['accuracy'].update(accuracy.item(), query_images.size(0))
                
                except Exception as e:
                    printM(f"Error in validation batch {batch_idx}: {e}")
                    continue
                
                # Update progress bar
                bar.suffix = (
                    f'Epoch: {epoch} | '
                    f'Batch: {batch_idx+1}/{n_episodes} | '
                    f'Loss: {self.val_metrics["loss"].avg:.4f} | '
                    f'KP Error: {self.val_metrics["keypoint_error"].avg:.4f} | '
                    f'Acc: {self.val_metrics["accuracy"].avg:.4f}'
                )
                bar.next()
        
        bar.finish()
        
        return {k: v.avg for k, v in self.val_metrics.items()}
    
    def train(
        self,
        num_epochs: int,
        save_best: bool = True,
        early_stopping_patience: Optional[int] = None
    ) -> Dict[str, List[float]]:
        """Full training loop"""
        printM(f"Starting training for {num_epochs} epochs")
        
        train_history = {k: [] for k in self.train_metrics.keys()}
        val_history = {k: [] for k in self.val_metrics.keys()}
        
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # Training
            train_metrics = self.train_epoch(epoch)
            for k, v in train_metrics.items():
                train_history[k].append(v)
            
            # Validation
            val_metrics = self.validate_epoch(epoch)
            for k, v in val_metrics.items():
                val_history[k].append(v)
            
            # Save checkpoint
            if epoch % self.save_interval == 0:
                self.save_checkpoint(epoch, train_metrics, val_metrics)
            
            # Save best model
            if save_best and val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.save_checkpoint(epoch, train_metrics, val_metrics, is_best=True)
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if early_stopping_patience is not None and patience_counter >= early_stopping_patience:
                printM(f"Early stopping after {epoch + 1} epochs")
                break
        
        printM("Training completed!")
        
        return {
            'train': train_history,
            'val': val_history
        }
    
    def _compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute training loss"""
        total_loss = 0.0
        
        # Keypoint prediction loss
        if 'keypoints' in predictions:
            keypoint_loss = nn.MSELoss()(predictions['keypoints'], targets)
            total_loss += keypoint_loss
        
        # Confidence loss (if available)
        if 'confidence' in predictions:
            confidence_loss = nn.BCELoss()(predictions['confidence'], torch.ones_like(predictions['confidence']))
            total_loss += 0.1 * confidence_loss
        
        # Distance loss (if available)
        if 'distances' in predictions:
            # Encourage correct class to have lower distances
            # This is a simplified version
            distance_loss = torch.mean(predictions['distances'])
            total_loss += 0.01 * distance_loss
        
        return total_loss
    
    def _compute_keypoint_error(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute keypoint localization error"""
        if predictions.shape != targets.shape:
            printM(f"Shape mismatch: pred {predictions.shape} vs target {targets.shape}")
            return torch.tensor(0.0, device=predictions.device)
        
        # Euclidean distance
        error = torch.norm(predictions - targets, dim=-1)  # [N_query, nkpts]
        return torch.mean(error)
    
    def _compute_accuracy(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute prediction accuracy (simplified)"""
        if 'confidence' not in predictions:
            return torch.tensor(0.0, device=predictions['keypoints'].device)
        
        # Simple accuracy based on confidence threshold
        confidence_threshold = 0.5
        accurate_predictions = (predictions['confidence'] > confidence_threshold).float()
        
        # For now, return mean confidence as "accuracy"
        return torch.mean(predictions['confidence'])
    
    def save_checkpoint(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        is_best: bool = False
    ):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'best_val_loss': self.best_val_loss,
            'global_step': self.global_step
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.save_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
            printM(f"Saved best model to {best_path}")
        
        printM(f"Saved checkpoint to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        if not os.path.exists(checkpoint_path):
            printM(f"Checkpoint not found: {checkpoint_path}")
            return
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        
        printM(f"Loaded checkpoint from {checkpoint_path}")
        printM(f"Resuming from epoch {self.current_epoch}")


class MetaLearningTrainer(FewShotTrainer):
    """
    Extended trainer for meta-learning approaches like MAML
    """
    
    def __init__(self, model: MetaLearningFSKD, *args, **kwargs):
        super().__init__(model, *args, **kwargs)
        self.meta_model = model
        
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Meta-learning training epoch"""
        printM(f"\n=== Meta-Learning Epoch {epoch} Training ===")
        
        # Reset metrics
        for metric in self.train_metrics.values():
            metric.reset()
        
        self.meta_model.train()
        
        n_episodes = len(self.train_loader)
        if n_episodes == 0:
            return {k: 0.0 for k in self.train_metrics.keys()}
        
        bar = Bar('Meta-Training', max=n_episodes)
        
        for batch_idx, episode_batch in enumerate(self.train_loader):
            support_images = episode_batch['support_images'].to(self.device)
            support_keypoints = episode_batch['support_keypoints'].to(self.device)
            query_images = episode_batch['query_images'].to(self.device)
            query_keypoints = episode_batch['query_keypoints'].to(self.device)
            
            # Skip empty batches
            if support_images.numel() == 0 or query_images.numel() == 0:
                continue
            
            try:
                # Meta-learning forward pass
                meta_result = self.meta_model.maml_forward(
                    support_images=support_images,
                    support_keypoints=support_keypoints,
                    query_images=query_images,
                    return_loss=True
                )
                
                # Compute meta loss
                meta_loss = self.meta_model.compute_meta_loss(
                    support_predictions=meta_result.get('support_predictions', {}),
                    query_predictions=meta_result['query_predictions'],
                    support_keypoints=support_keypoints,
                    query_keypoints=query_keypoints
                )
                
                # Meta-update
                self.optimizer.zero_grad()
                meta_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.meta_model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                # Update metrics
                self.train_metrics['loss'].update(meta_loss.item(), support_images.size(0))
                
                if batch_idx % self.log_interval == 0:
                    printM(f"Epoch {epoch} [{batch_idx}/{n_episodes}] Meta-Loss: {meta_loss.item():.4f}")
                
                self.global_step += 1
                
            except Exception as e:
                printM(f"Error in meta-training batch {batch_idx}: {e}")
                continue
            
            bar.next()
        
        bar.finish()
        self.current_epoch = epoch
        
        return {k: v.avg for k, v in self.train_metrics.items()}
