import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="codes for ddp")
    parser.add_argument(
        '--cfg', 
		help='Path to the configuration file',
		default='conf/deep_pose.yaml',
		type=str,
	)
    parser.add_argument(
		'--model', 
		help='Which model to use [DeepPose]'
	)
    parser.add_argument(
		'--loadModel', 
		help='Path to the model to load'
	)   
    return parser.parse_known_args()[0]
