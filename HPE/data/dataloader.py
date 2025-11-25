from torch.utils.data import DataLoader

def dataloader(args, split):
	return DataLoader(
		
		batch_size = args.data_loader_size,
		shuffle = args.shuffle if split=='train' else False,
		pin_memory = not(args.dont_pin_memory),
		num_workers = args.nThreads
	)