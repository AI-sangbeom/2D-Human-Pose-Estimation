from yacs.config import CfgNode as CN

C = CN()

C.gpus = 0
C.saveDir = ''

C.model = CN()
C.model.name = 'DeepPose'
C.model.backbone = 'resnet50'
C.model.nkpts = 17
C.model.input_res = 224
C.model.output_res = 32
C.model.load_model = ''
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
C.train.print_freq = 10
C.save_freq = 5
C.train.dropLR = 5
C.train.dropMag = 0.7

C.valid = CN()
C.valid.val_interval = 3
C.valid.save_interval = 5

C.dataset = CN()
C.dataset.name = 'MPII'
C.dataset.data_dir = './data/MPII/images'
C.dataset.train_split = 'train'
C.dataset.val_split = 'valid'
C.dataset.num_workers = 4

def update_config(cfg, args):
    cfg.defrost()
    cfg.merge_from_file(args.cfg)
    cfg.freeze()