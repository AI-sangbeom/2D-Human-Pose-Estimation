import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="codes for ddp")
    parser.add_argument(
        '--cfg', 
		help='Path to the configuration file',
		default='configs/method/deep_pose.yaml',
		type=str,
	)
    parser.add_argument(
		'--test', 
		help='model test',
        default=False,
	)
    parser.add_argument(
        '--gpus',
        help='GPUs to use, e.g. 0,1,2,3',
        default=None,
        type=tuple,
	)
    parser.add_argument(
		'--checkpoint', 
		help='Path to the checkpoint',
        default=None,
		type=str,
	)   
    return parser.parse_known_args()[0]
