from utils import line
from yacs.config import CfgNode as CN

C = CN()
C.title = 'HPE_Experiment'
C.description = "Few-shot keypoint detection with DINOv3 small backbone"
C.seed = 42
C.use_deterministic = True
C.gpus = 0, 
C.saveDir = ''

C.model = CN()
C.model.name = 'DeepPose'
C.model.backbone = 'resnet50'
C.model.iou_threshold = 0.5
C.model.nkpts = 17
C.model.checkpoint = ''
C.model.pretrained = True

# Few-shot learning parameters
C.model.n_way = 5
C.model.k_shot = 1
C.model.temperature = 1.0
C.model.fusion_method = 'cross_attention'
C.model.use_hierarchical_prototypes = False
C.model.freeze_backbone = False
C.model.dropout = 0.1

# Train configuration
C.train = CN()
C.train.batch_size = 32
C.train.num_epochs = 100
C.train.mini_batch_count = 1
C.train.optimizer = 'SGD'
C.train.shuffle = True
C.train.lr = 2.5e-4
C.train.lr_step = 30
C.train.lr_gamma = 0.1
C.train.momentum = 0.9
C.train.weight_decay = 5e-4
C.train.save_freq = 10
C.train.dropLR = 5
C.train.dropMag = 0.7

# Few-shot training parameters
C.train.n_way = 5
C.train.k_shot = 1
C.train.n_queries = 1
C.train.num_episodes = 10000
C.train.meta_lr = 1e-3
C.train.grad_clip_norm = 1.0
C.train.update_steps = 5
C.train.adaptation_lr = 0.01

# Validation and evaluation
C.valid = CN()
C.valid.val_interval = 3
C.valid.save_interval = 5
C.valid.val_episodes = 1000

# Keep existing valid settings for backward compatibility
C.valid.val_interval = 5
C.valid.save_interval = 5

C.test = CN()
C.test.batch_size = 32
C.test.shuffle = False
C.test.test_episodes = 1000
C.test.n_way = 5
C.test.k_shot = 1
C.test.n_queries = 5

# Data configuration
C.data = CN()
C.data.dataset = 'MPII'
C.data.data_dir = './data/MPII/images'
C.data.train_split = 'train'
C.data.val_split = 'valid'
C.data.test_split = 'test'

# Episode generation
C.data.episode_length = 100
C.data.class_sampling = 'random'

# Data Loading
C.data.num_workers = 4
C.data.pin_memory = True
C.data.drop_last = False

# Loss function configuration
C.loss = CN()
C.loss.keypoint_weight = 1.0
C.loss.confidence_weight = 0.1
C.loss.distance_weight = 0.01
C.loss.temperature = 1.0

C.metric = CN()
C.metric.classify = 'CMet'
C.metric.detect = 'DMet'
C.metric.pose = 'PMet'

C.logging = CN()
C.logging.level = 'INFO'
C.logging.save_dir = ''
C.logging.tensorboard_dir = ''
C.logging.checkpoint_dir = ''

C.hardware = CN()
C.hardware.device = 'cuda'
C.hardware.mixed_precision = False

from utils import printM

@line
def update_config(cfg, args):
    cfg.defrost()
    cfg.merge_from_file(args.cfg)
    if args.gpus is not None:
        gpus = args.gpus.replace(',', '')
        gpus = tuple(int(gpu) for gpu in gpus)
        cfg.gpus = gpus
    if args.ckpt is not None:
        cfg.model.checkpoint = args.ckpt
    cfg.freeze()
    printM(f"[Configuration]\n")
    printM(cfg)
