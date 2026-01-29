import os 
import torch
import torch.nn as nn
from tqdm import tqdm 
from pose.utils import DISTRIBUTED, MASTER_RANK, printM, printT, printS, colored_msg
from pose.utils.dist import DDPManager, set_seed


class Trainer:
    def __init__(
            self, 
            cfg, 
            model:nn.Module,
            trainloader, 
            validloader,
            optimizer, 
            lr_scheduler, 
            loss_fn, 
            ddp_manager: DDPManager,
            metric=None,
            use_scalar=False,
        ):

        set_seed(cfg.seed, cfg.use_deterministic)
        self.cfg = cfg
        self.task = cfg.task
        self.ddp_manager = ddp_manager
        self.device = self.ddp_manager.device
        self.model = model
        self.trainloader = trainloader
        self.validloader = validloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss_fn = loss_fn
        self.metric = metric
        self.scaler = torch.amp.GradScaler(enabled=True) if use_scalar else None

        self.output_path = cfg.trainer.save_path if hasattr(cfg.trainer, 'save_path') else "weights"
        self.output_path = os.path.join('output', self.output_path)
        os.makedirs(self.output_path, exist_ok=True)
        self.metric_score()
    
    def metric_score(self):
        self.epoch_loss = 0.0
        self.best_loss = float('inf')

    def iter_one_epoch(self, epoch, dataloader=None, train=True):
        pbar = tqdm(dataloader, 
                    total=len(dataloader),
                    dynamic_ncols=True,
                    mininterval=0.5,) if MASTER_RANK else dataloader
        self.epoch_loss = 0.0
        for images, targets in pbar:
            imgs = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            imgs = imgs.float() / 255.0
            self.optimizer.zero_grad()

            with torch.amp.autocast(device_type='cuda', enabled=self.scaler is not None):
                preds = self.model(imgs)
                loss, loss_items = self.loss_fn(preds, targets)

            if train:
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer) 
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()
            
            current_loss = loss.item()
            l_cls, l_kpt, l_obj = self.loss_fn.add_loss(loss_items)
            if train:
                self.epoch_loss += current_loss
            if MASTER_RANK:
                message = f"{epoch+1} epoch | loss: {current_loss:.4f} | obj: {l_obj:.4f} | cls: {l_cls:.4f} | kpt: {l_kpt:.4f}"
                pbar.desc = f" {colored_msg('[TRAIN]', 'green')} {message}"
            

    def train(self):
        epochs = self.cfg.trainer.epochs
        self.model.train()
        warmup_epochs = self.cfg.lr_scheduler.warmup_epochs
        for epoch in range(epochs):
            printM()
            self.loss_fn.set_train_loss()
            self.warmup(epoch, warmup_epochs) if epoch < warmup_epochs else None
            self.iter_one_epoch(epoch, self.trainloader)
            self.lr_scheduler.step()
            num_batches = max(1, len(self.trainloader))
            avg_loss = self.epoch_loss / num_batches
            avg_cls = self.loss_fn.cls_loss_sum / num_batches
            avg_kpt = self.loss_fn.kpt_loss_sum / num_batches
            avg_obj = self.loss_fn.obj_loss_sum / num_batches
            printT(f"  total | loss: {avg_loss:.4f} | obj: {avg_obj:.4f} | cls: {avg_cls:.4f} | kpt: {avg_kpt:.4f}")
            self.save_checkpoint(epoch, avg_loss)

            # if epoch % self.cfg.trainer.valid_term==0:
            #     self.validate(epoch)

    def warmup(self, epoch, warmup_epochs):
        warmup_factor = (epoch + 1) / warmup_epochs
        for param_group in self.optimizer.param_groups:
            if 'initial_lr' in param_group:
                param_group['lr'] = param_group['initial_lr'] * warmup_factor
            else:
                param_group['lr'] = param_group['lr'] * warmup_factor


    def validate(self, epoch):
        self.model.eval()
        self.loss_fn.set_train_loss()
        self.iter_one_epoch(epoch=epoch, dataloader=self.validloader, train=False)
        num_batches = max(1, len(self.validloader))
        avg_loss = self.epoch_loss / num_batches
        avg_cls = self.loss_fn.cls_loss_sum / num_batches
        avg_kpt = self.loss_fn.kpt_loss_sum / num_batches
        avg_obj = self.loss_fn.obj_loss_sum / num_batches
        printT(f"[VALID] | loss: {avg_loss:.4f} | obj: {avg_obj:.4f} | cls: {avg_cls:.4f} | kpt: {avg_kpt:.4f}")

    def save_checkpoint(self, epoch, avg_loss):
        state_dict = self.model.module.state_dict() if DISTRIBUTED else self.model.state_dict()
        if (epoch + 1) % self.cfg.trainer.save_term == 0 or (epoch + 1) == self.cfg.trainer.epochs:
            torch.save(state_dict, f"{self.output_path}/pose_dino_epoch_{epoch+1}.pt")
            printS(f"Saved checkpoint: {self.output_path}/pose_dino_epoch_{epoch+1}.pt")
        
        if avg_loss < self.best_loss:
            self.best_loss = avg_loss
            torch.save(state_dict, f"{self.output_path}/best.pt")
            printS(f"Saved best model (loss: {self.best_loss:.4f})")

    def load_checkpoint(self, path):
        if path and os.path.exists(path):
            param = torch.load(path, map_location=self.device)
            self.model.load_state_dict(param)
            printS(f"Loaded checkpoint from {path}")

    def cleanup(self):
        self.ddp_manager.cleanup()

    