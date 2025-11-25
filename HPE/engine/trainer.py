from core.builder import Builder
from utils import DDPManager, set_seed

class Trainer(DDPManager):
    def __init__(self, cfgs, builder: Builder, logFile: str):
        super(Trainer, self).__init__(cfgs.gpus)
        set_seed(cfgs.seed)
        self.target_gpu_ids = cfgs.gpus  # 예: [2, 4]
        self.logFile = logFile
        self.cfgs = cfgs

    def setup(self, builder: Builder):
        self.model = builder.model()
        self.criterion = builder.loss()
        self.metric = builder.metric()
        self.optimizer = builder.optimizer(self.model)
        self.train_dataloader, self.valid_dataloader = builder.dataloader()