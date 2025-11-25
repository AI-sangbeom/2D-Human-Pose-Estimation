from data.dataloader import dataloader

class Builder(object):
	"""docstring for Builder"""
	def __init__(self, cfgs, args):
		super(Builder, self).__init__()
		self.cfgs = cfgs
		self.test = args.test
		# self.train_mode = cfgs.train_mode

	def model(self):
		pass

	def loss(self):
		pass

	def metric(self):
		pass
			
	def optimizer(self, model):
		pass

	def dataloader(self):
		pass

	def epoch(self):
		pass