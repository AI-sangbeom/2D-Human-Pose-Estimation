import torch
import models
import core.metric as metric
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
		model_name = self.mcfg.name
		assert model_name in models.__all__, f'Unknown model : "{model_name}"\n        Available models : {models.__all__}'
		model_builder = getattr(models, model_name)
		model = model_builder(self.mcfg.nkpts, self.mcfg.backbone, self.mcfg.pretrained)
		
		if self.mcfg.checkpoint is not None:
			state_dict = torch.load(self.mcfg.checkpoint)
			model.load_state_dict(state_dict)
		return model

	def loss(self):
		pass

	def metric(self):
		metric_name = self.cfg.valid.metric
		assert metric.__all__, f'Unknown metric : "{metric_name}"\n        Available metrics : {metric.__all__}'
		metric = getattr(metric, self.cfg.valid.metric)
		pass
			
	def optimizer(self, model):
		pass

	def dataloader(self):
		return None, None

	def epoch(self):
		pass