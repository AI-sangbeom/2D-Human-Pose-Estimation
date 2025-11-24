import torch
import models
import losses
import metrics
import dataloaders


class Builder(object):
	"""docstring for Builder"""
	def __init__(self, args):
		super(Builder, self).__init__()
		self.args = args
		if args.loadModel is not None:
			self.states = torch.load(args.loadModel)
		else:
			self.states = None

	def Model(self):
		ModelBuilder = getattr(models, self.args.model)
		if self.args.model == 'DeepPose':
			Model = ModelBuilder(self.args.nJoints, self.args.baseName)
		else:
			assert('Not Implemented Yet!!!')
		if self.states is not None:
			Model.load_state_dict(self.states['model_state'])
		return Model

	def Loss(self):
		instance = losses.Loss(self.args)
		return getattr(instance, self.args.model)

	def Metric(self):
		PCKhinstance = metrics.PCKh(self.args)
		PCKinstance = metrics.PCK(self.args)
		if self.args.dataset=='MPII':
			return {'PCK' : getattr(PCKinstance, self.args.model), 'PCKh' : getattr(PCKhinstance, self.args.model)}         
		if self.args.dataset=='COCO':
			return {'PCK' : getattr(PCKinstance, self.args.model)}
			
	def Optimizer(self, Model):
		TrainableParams = filter(lambda p: p.requires_grad, Model.parameters())
		Optimizer = getattr(torch.optim, self.args.optimizer_type)(TrainableParams, lr = self.args.LR, alpha = 0.99, eps = 1e-8)
		if self.states is not None and self.args.loadOptim:
			Optimizer.load_state_dict(self.states['optimizer_state'])
			if self.args.dropPreLoaded:
				for i,_ in enumerate(Optimizer.param_groups):
					Optimizer.param_groups[i]['lr'] /= self.args.dropMagPreLoaded
		return Optimizer

	def DataLoaders(self):
		return dataloaders.ImageLoader(self.args, 'train'), dataloaders.ImageLoader(self.args, 'val')

	def Epoch(self):
		Epoch = 1
		if self.states is not None and self.args.loadEpoch:
			Epoch = self.states['epoch']
		return Epoch