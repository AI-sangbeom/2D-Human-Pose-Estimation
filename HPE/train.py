import os 
import torch 
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from models.pose import DINOv3Pose
from loss import *
from dataset import * 

hyp = {
    'lr_cls': 1e-2,      # [설정] Classification Head 학습률
    'lr_kpt': 1e-3,      # [설정] Keypoint/Backbone 학습률
    'epochs': 100,
    'batch_size': 32,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'weight_decay': 0.05,
    'warmup_epochs': 3
}

def train(ckps=None):
    device = hyp['device']
    
    print(f"Initializing DINOv3Pose on {device}...")
    model = DINOv3Pose(backbone='dinov3_convnext_small', nkpts=(4, 3), ncls=10, device=device)
    
    if ckps and os.path.exists(ckps):
        param = torch.load(ckps, map_location=device)
        model.load_state_dict(param)
        print(f"Loaded checkpoint from {ckps}")
    
    model = model.to(device)
    model.train()

    cls_params = []
    kpt_params = []
    
    print("Separating parameters...")
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        # 이름에 'cls'나 'class'가 들어가면 분류 헤드로 간주
        # (실제 모델의 레이어 이름을 확인해보시고 키워드를 조정하세요)
        if 'cv3' in name or 'cv4' in name:
            cls_params.append(param)
            # print(f"  [CLS Group] {name}") # 확인용 출력
        else:
            kpt_params.append(param)
    
    # 그룹별로 dict를 만들어 optimizer에 전달
    optimizer = optim.AdamW([
        {'params': cls_params, 'lr': hyp['lr_cls']},  # 1e-2
        {'params': kpt_params, 'lr': hyp['lr_kpt']}   # 1e-3
    ], weight_decay=hyp['weight_decay'])

    # Scheduler는 optimizer에 설정된 초기 lr을 기준으로 작동합니다.
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=hyp['epochs'], eta_min=1e-5)

    # -----------------------------------------------------------
    
    print("Loading Data...")
    base_folder = '/media/otter/otterHD/AXData/TotalAX/'
    train_img_dir = os.path.join(base_folder, "images")
    train_label_dir = os.path.join(base_folder, 'labels')
    
    batch_size = hyp['batch_size']
    img_size = 640
    
    dataset = YoloPoseDataset(train_img_dir, train_label_dir, img_size=img_size, nkpts=4)
    
    dataloader = DataLoader(dataset, 
                            batch_size=batch_size, 
                            shuffle=True, 
                            collate_fn=YoloPoseDataset.collate_fn, 
                            num_workers=8,
                            prefetch_factor=4,
                            persistent_workers=True, # HDD라면 필수!
                            pin_memory=True)

    print(f"Dataset size: {len(dataset)}")
    print("Setting up Loss...")
    criterion = ComputeLoss(model)
    
    scaler = torch.amp.GradScaler(enabled=True)
    best_loss = float('inf')
    
    for epoch in range(hyp['epochs']):
        model.train()
        
        # -----------------------------------------------------------
        # [수정 2] Warmup 로직 수정 (그룹별 초기 LR 비율 유지)
        # -----------------------------------------------------------
        if epoch < hyp['warmup_epochs']:
            # 스케줄러가 optimizer에 저장해둔 'initial_lr'을 가져와서 비율대로 줄임
            warmup_factor = (epoch + 1) / hyp['warmup_epochs']
            for param_group in optimizer.param_groups:
                # param_group['initial_lr']은 scheduler 선언 시 자동 생성됨
                if 'initial_lr' in param_group:
                    param_group['lr'] = param_group['initial_lr'] * warmup_factor
                else:
                    # 혹시 없으면 현재 lr 기준으로 (보통은 위 if문으로 들어감)
                    param_group['lr'] = param_group['lr'] * warmup_factor
        
        # 현재 LR 출력용 (첫 번째 그룹과 두 번째 그룹 확인)
        pbar = tqdm(enumerate(dataloader), 
                    total=len(dataloader),
                    dynamic_ncols=True,
                    mininterval=0.5)
        
        epoch_loss = 0
        cls_loss_sum = 0
        kpt_loss_sum = 0
        obj_loss_sum = 0
        
        for i, (imgs, targets) in pbar:
            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            imgs = imgs.float() / 255.0
            
            optimizer.zero_grad()
            
            with torch.amp.autocast(device_type='cuda', enabled=True):
                preds = model(imgs)
                loss, loss_items = criterion(preds, targets)
            
            # -----------------------------------------------------------
            # [수정 3] Gradient Clipping 위치 수정 (중요!)
            # 순서: Backward -> Unscale -> Clip -> Step
            # -----------------------------------------------------------
            scaler.scale(loss).backward()
            
            # Gradient Clipping은 unscale된 상태에서 해야 올바르게 작동함
            scaler.unscale_(optimizer) 
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            # Logging
            current_loss = loss.item()
            l_cls = loss_items[0].item() if torch.is_tensor(loss_items[0]) else loss_items[0]
            l_kpt = loss_items[1].item() if torch.is_tensor(loss_items[1]) else loss_items[1]
            l_obj = loss_items[2].item() if torch.is_tensor(loss_items[2]) else loss_items[2]

            epoch_loss += current_loss
            cls_loss_sum += l_cls
            kpt_loss_sum += l_kpt
            obj_loss_sum += l_obj
            
            pbar.set_postfix({
                "loss": f"{current_loss:.4f}",
                "cls": f"{l_cls:.4f}",
                "kpt": f"{l_kpt:.4f}",
                "obj": f"{l_obj:.4f}",
            })
        
        # Warmup 이후에는 스케줄러가 LR 조절
        if epoch >= hyp['warmup_epochs']:
            scheduler.step()
        
        num_batches = max(1, len(dataloader))
        avg_loss = epoch_loss / num_batches
        avg_cls = cls_loss_sum / num_batches
        avg_kpt = kpt_loss_sum / num_batches
        avg_obj = obj_loss_sum / num_batches
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Total: {avg_loss:.4f} | Obj: {avg_obj:.4f} | Cls: {avg_cls:.4f} | KPT: {avg_kpt:.4f}\n")
        
        os.makedirs("weights", exist_ok=True)
        if (epoch + 1) % 10 == 0 or (epoch + 1) == hyp['epochs']:
            torch.save(model.state_dict(), f"weights/pose_dino_epoch_{epoch+1}.pt")
            print(f"Saved checkpoint: weights/pose_dino_epoch_{epoch+1}.pt")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "weights/best.pt")
            print(f"Saved best model (loss: {best_loss:.4f})")

    print("Training Completed!")

if __name__ == '__main__':
    train()