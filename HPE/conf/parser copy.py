import os
import datetime
import argparse

def ensure_dir(path):
	if path is not None:
		if not os.path.exists(path):
			os.makedirs(path)

class Parser():
	"""docstring for opts"""
	def __init__(self):
		super(Parser, self).__init__()
		self.parser = argparse.ArgumentParser(description='2D Human Pose Estimation')
		
	def init(self):
		self.parser.add('-test', action='store_true', help='Run only the validation epoch on the test dataset')
		self.parser.add('-usedouble', action='store_true', help='Change Default tensor type to double')
		self.parser.add('-DEBUG', action='store_true', help='To run in debug mode (visualize heatmaps, skeletons (ground truth and predicted)')
		self.parser.add('-dont_pin_memory', default=0, help='Whether to pin memory to gpu or not from the dataloader')

		self.parser.add('-DataConfig', required=True, is_config_file=True, help='Path to config file')
		self.parser.add('-ModelConfig', required=True, is_config_file=True, help='Path to config file')

		self.parser.add('--expDir', help='Experiment-Directory')
		self.parser.add('--expID', help='Experiment-ID')

		self.parser.add('--visdom', type=int, help='Support for visdom (currently unavailable :( ')


######## Model Name and Load Checkpoint Parameters
		self.parser.add('--model', help='Which model to use [DeepPose] [ChainedPredictions] [StackedHourGlass] [PyraNet] [PoseAttention]')

		self.parser.add('--loadModel', help='Path to the model to load')
		self.parser.add('--loadOptim', type=int, help='Whether to load Optimizer Parameters')
		self.parser.add('--dropPreLoaded', type=int, help='Whether to Drop Learning Rate of Loaded Optimizer')
		self.parser.add('--dropMagPreLoaded', type=float, help='How much learning rate to be dropped from the optimizer')
		self.parser.add('--loadEpoch', type=int, help='Whether to load Epoch number')


######## Dataloader Parameters
		self.parser.add('--TargetType', help='TargetType for the dataloader [(direct) : Targets for Direct Regression] [(heatmap) : Targets for Heatmap Regression]')
		self.parser.add('--maxTranslate', type=float, help='Maximum translation as a percentage of the image width')
		self.parser.add('--maxScale', type=float, help='How much to scale the image or zoom in (for augmentation)')
		self.parser.add('--maxRotate', type=float, help='Maximum angle of rotation on either side (for augmentation)')
		self.parser.add('--dataDir', help='Directory for the data')
		self.parser.add('--imageRes', type=int, help='Size of Image Loaded')
		self.parser.add('--inputRes', type=int, help='Size of input to the network')
		self.parser.add('--outputRes', type=int, help='Size of output in case of heatmap based networks')
		self.parser.add('--hmGauss', type=int, help='Heatmap Gaussian Size')

		self.parser.add('--nJoints', type=int, help='Number of Joints to learn from dataset')

######## Network Parameters

		#### ChainedPredictions
		self.parser.add('--modelName', help='Network Parameter')
		self.parser.add('--hhKernel', type=int, help='Network Parameter')
		self.parser.add('--ohKernel', type=int, help='Network Parameter')

		#### DeepPose
		self.parser.add('--baseName', help='Network Parameter')

		#### StackedHourGlass | PyraNet | PoseAttention
		self.parser.add('--nChannels', type=int, help='Network Parameter')
		self.parser.add('--nStack', type=int, help='Network Parameter')
		self.parser.add('--nModules', type=int, help='Network Parameter')
		self.parser.add('--nReductions', type=int, help='Network Parameter')

		#### PyraNet
		self.parser.add('--baseWidth', type=int, help='Network Parameter')
		self.parser.add('--cardinality', type=int, help='Network Parameter')

		#### PoseAttention
		self.parser.add('--LRNSize', type=int, help='Network Parameter')
		self.parser.add('--IterSize', type=int, help='Network Parameter')


######## DataLoader Parameters
		self.parser.add('--dataset', help='MPII or COCO')
		self.parser.add('--shuffle', type=int, help='Shuffle the data during training')
		self.parser.add('--nThreads', type=int, help='How many threads to use for Dataloader')

		self.parser.add('--data_loader_size', type=int, help='Batch Size for DataLoader')
		self.parser.add('--mini_batch_count', type=int, help='After how many mini batches to run backprop')

		self.parser.add('--valInterval', type=int, help='After how many train epoch to run a val epoch')
		self.parser.add('--saveInterval', type=int, help='After how many train epochs to save model')

		self.parser.add('--gpuid', type=int, help='GPU ID for the model')
		self.parser.add('--nEpochs', type=int, help='Number of epochs to train')

		self.parser.add('--optimizer_type', help='Which optimizer to use in DataLoader')
		self.parser.add('--optimizer_pars', action='append' , help='parameters for the optimizer')
		self.parser.add('--LR', type=float, help='Learning rate for the base resnet')

		self.parser.add('--dropLR', type=int, help='Drop LR after how many epochs')
		self.parser.add('--dropMag', type=float, help='Drop LR magnitude')

		self.parser.add('--worldCoors', help='World Coordinates file path (only for MPII)')
		self.parser.add('--headSize', help='head Size file path (only for MPII)')

	def parse(self):
		self.init()
		self.args = self.parser.parse_args()
		if self.args.DEBUG:
			self.args.data_loader_size = 1
			self.args.shuffle = 0

		self.args.saveDir = os.path.join(os.path.join(self.args.expDir, self.args.expID), os.path.join(self.args.model, 'logs_{}'.format(datetime.datetime.now().isoformat())))
		self.args.saveDir = os.path.join(self.args.expDir, self.args.model, self.args.expID, 'logs_{}'.format(datetime.datetime.now().isoformat()))
		ensure_dir(self.args.saveDir)

		####### Write All Opts
		args = dict((name, getattr(self.args, name)) for name in dir(self.args)
					if not name.startswith('_'))

		file_name = os.path.join(self.args.saveDir, 'args.txt')
		with open(file_name, 'wt') as opt_file:
			opt_file.write('==> Args:\n')
			for k, v in sorted(args.items()):
				opt_file.write("%s: %s\n"%(str(k), str(v)))

		return self.args