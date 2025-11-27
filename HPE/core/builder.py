import torch
import models
from data.dataloader import dataloader
from utils import printE

class Builder(object):
	"""docstring for Builder"""
	def __init__(self, cfg, args):
		super(Builder, self).__init__()
		self.cfg = cfg
		self.test = args.test
		self.mcfg = self.cfg.model

	def model(self):
		assert self.mcfg.name in models.__all__, printE(f'Unknown model : "{self.mcfg.name}"\n        Available models : {models.__all__}')
		model_builder = getattr(models, self.mcfg.name)
		model = model_builder(self.mcfg.nkpts, self.mcfg.backbone, self.mcfg.pretrained)
		
		if self.mcfg.checkpoint is not None:
			state_dict = torch.load(self.mcfg.checkpoint)
			model.load_state_dict(state_dict)
		return model

	def loss(self):
		pass

	def metric(self):
		pass
			
	def optimizer(self, model):
		pass

	def dataloader(self):
		return None, None

	def epoch(self):
		pass