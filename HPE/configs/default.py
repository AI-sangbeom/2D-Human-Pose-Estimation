from utils import line
from yacs.config import CfgNode as CN

C = CN()
C.title = 'HPE_Experiment'

C.seed = 42
C.use_deterministic = True
C.gpus = 0, 
C.num_workers = 4
C.saveDir = ''

C.model = CN()
C.model.name = 'DeepPose'
C.model.backbone = 'resnet50'
C.model.iou_threshold = 0.5
C.model.nkpts = 17
C.model.checkpoint = ''
C.model.pretrained = True

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

C.valid = CN()
C.valid.val_interval = 3
C.valid.save_interval = 5
C.valid.metric = 'mAP'

C.test = CN()
C.test.batch_size = 32
C.test.shuffle = False

C.data = CN()
C.data.name = 'MPII'
C.data.data_dir = './data/MPII/images'
C.data.train_split = 'train'
C.data.val_split = 'valid'
C.data.ncls = 10
C.data.topk = 5

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
