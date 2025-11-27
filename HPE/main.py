import warnings
warnings.filterwarnings('ignore', category=UserWarning)
import os 
import torch.distributed as distributed
from utils import *
from configs import *
from core.builder import Builder
from engine.trainer import Trainer

def main(args):
    # Initialize configuration
    try:
        update_config(cfg, args)
        logFile = os.path.join(save_dir(cfg), 'log.txt')
        builder = Builder(cfg, args)
        trainer = Trainer(cfg, builder, logFile)
        trainer.cleanup()
    except Exception as e:
        printE(e)
    finally:
        try:
            distributed.destroy_process_group()
        except:
            pass 

if __name__ == '__main__':
    main(parse_args())
