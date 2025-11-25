import os 
from utils import *
from configs import *
from core.builder import Builder
from engine.trainer import Trainer

# Initialize configuration
args = parse_args()
update_config(cfg, args)
logFile = os.path.join(cfg.saveDir, 'log.txt')
builder = Builder(cfg, args)
trainer = Trainer(cfg, builder, logFile)