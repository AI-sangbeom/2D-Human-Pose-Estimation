import os
import datetime
import argparse

class Parser:
	def __init__(self):
		super(Parser, self).__init__()
		self.parser = argparse.ArgumentParser(description='2D Human Pose Estimation')
			
	def update_config(self, cfg, args):
		cfg.defrost()
		cfg.merge_from_file(args.cfg)
		cfg.freeze()
		
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

	def init(self):
		self.parser.add_argument(
			'--cfg', 
			help='Path to the configuration file',
			default='conf/deep_pose.yaml',
			type=str,
        )
		self.parser.add_argument(
			'--model', 
			help='Which model to use [DeepPose]'
        )
		self.parser.add_argument(
			'--loadModel', 
			help='Path to the model to load'
        )

def ensure_dir(path):
	if path is not None:
		if not os.path.exists(path):
			os.makedirs(path)