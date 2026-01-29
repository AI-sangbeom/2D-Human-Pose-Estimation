from yacs.config import CfgNode as CN

C = CN()
C.title = "Pose Estimation with DINOv3"
C.task = 'pose'
C.seed = 42
C.use_deterministic = True
C.gpus = 0,

C.model = CN()
C.model.model_name = 'custom_dinov3convnext'
C.model.nkpts = (4, 3)
C.model.ncls = 7
C.model.backbone_name = 'dinov3_convnext_base'
C.model.backbone_ckps = None
C.model.finetuning = True

C.dataset = CN()
C.dataset.img_size = 512
C.dataset.dataset = 'yolo_pose'
C.dataset.train_dir = 'data/train'
C.dataset.valid_dir = 'data/valid'

C.dataloader = CN()
C.dataloader.batch_size = 16
C.dataloader.num_workers = 4
C.dataloader.pin_memory = True
C.dataloader.shuffle = True
C.dataloader.drop_last = True

C.trainer = CN()
C.trainer.epochs = 100
C.trainer.save_path = "weights"
C.trainer.valid_term = 5
C.trainer.save_term = 100//10


C.lr_scheduler = CN()
C.lr_scheduler.type = 'cosine'
C.lr_scheduler.warmup_epochs = 5
C.lr_scheduler.min_lr = 1e-6

C.optimizer = CN()
C.optimizer.type = 'adamw'
C.optimizer.lr = 1e-4
C.optimizer.weight_decay = 1e-2

