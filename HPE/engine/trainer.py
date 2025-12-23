import os 
import torch
from progress.bar import Bar
from core.builder import Builder
from utils import AverageMeter, DDPManager, set_seed, adjust_learning_rate, printM, MASTER_RANK

class Trainer(DDPManager):
    def __init__(self, cfg, builder: Builder, logFile: str):
        super(Trainer, self).__init__(cfg.gpus)
        set_seed(cfg.seed, cfg.use_deterministic)
        self.target_gpu_ids = cfg.gpus  # 예: [2, 4]
        self.logFile = logFile
        self.cfg = cfg
        self.ckps_folder = os.path.join(self.cfg.saveDir, 'checkpoints')
        self.best_train = 100
        self.setup(builder)

    def setup(self, builder: Builder):
        self.model = builder.model()
        self.criterion = builder.loss()
        self.metrics = builder.metric()
        self.optimizer = builder.optimizer(self.model)
        self.train_dataloader, self.valid_dataloader = builder.dataloader()

    def train(self, start_epoch, end_epoch):
        for epoch in range(start_epoch, end_epoch+1):
            log = self._epoch(self.train_dataloader, epoch, mode='train')

            if epoch%self.cfg.valid.val_interval==0:
                with torch.no_grad():
                    test = self._epoch(self.valid_dataloader, epoch, 'val')
                Writer = open(self.logFile, 'a')
                Writer.write(log + ' ' + test + '\n')
                Writer.close()
            else:
                Writer = open(self.logFile, 'a')
                Writer.write(log + '\n')
                Writer.close()

            if epoch%self.cfg.valid.save_interval==0:
                state = {
                    'epoch': epoch+1,
                    'model_state': self.model.state_dict(),
                    'optimizer_state' : self.optimizer.state_dict(),
                }
                path = os.path.join(self.ckps_folder, f'model_{epoch}.pt')
                torch.save(state, path)

            if self.best_train > self.loss.avg:
                self.best_train = self.loss.avg
                state = {
                    'epoch': epoch+1,
                    'model_state': self.model.state_dict(),
                    'optimizer_state' : self.optimizer.state_dict(),
                }
                path = os.path.join(self.ckps_folder, 'best.pt')
                torch.save(state, path)
                
            adjust_learning_rate(self.optimizer, epoch, self.cfg.dropLR, self.cfg.dropMag)
        loss_final = self._epoch(self.valid_dataloader, -1, 'val')
        printM(loss_final)

    def test(self):
        with torch.no_grad():
            self._epoch(self.valid_dataloader, -1, 'val')

    def init_epoch(self):
        self.loss = AverageMeter
        self.loss.reset()
        for key, value in self.metrics.items():
            setattr(self, key, AverageMeter())
        for key, value in self.metrics.items():
            getattr(self, key).reset()

    def _epoch(self, dataloader, epoch, mode):
        self.init_epoch()
        if mode == 'train':
            self.model.train()
        else:
            self.model.eval()
        nIters = len(dataloader)
        
        if MASTER_RANK:
            bar = Bar('==>', max=nIters) 

        for batch_idx, (data, target, meta1, meta2) in enumerate(dataloader):
            model = self.model.to(self.device)
            data = data.to(self.device, non_blocking=True).float()
            target = target.to(self.device, non_blocking=True).float()
            output = model(data)

            loss = self.criterion(output, target, meta1.to(self.device, non_blocking=True).float().unsqueeze(-1))
            self.loss.update(loss.item(), data.shape[0])
            self._eval_metrics(output, target, meta1, meta2, data.shape[0])

            if mode == 'train':
                loss.backward()
                if (batch_idx+1)%self.cfg.mini_batch_count==0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            if MASTER_RANK:
                Bar.suffix = mode + ' Epoch: [{0}][{1}/{2}]| Total: {total:} | ETA: {eta:} | Loss: {loss.avg:.6f} ({loss.val:.6f})'.format(epoch, batch_idx+1, nIters, total=bar.elapsed_td, eta=bar.eta_td, loss=self.loss) + self._print_metrics()
                bar.next()
        if MASTER_RANK:
            bar.finish()
        return '{:8f} '.format(self.loss.avg) + ' '.join(['{:4f}'.format(getattr(self, key).avg) for key,_ in self.metrics.items()])


    def _eval_metrics(self, output, target, meta1, meta2, batchsize):
        for key, value in self.metrics.items():
            value, count = value(output, target, meta1, meta2)
            getattr(self, key).update(value, count)
        return

    def _print_metrics(self):
        return ''.join([('| {0}: {metric.avg:.3f} ({metric.val:.3f}) '.format(key, metric=getattr(self, key))) for key, _ in self.metrics.items()])