import torch 
import os, glob, cv2
import numpy as np 
from torch.utils.data import Dataset

class YoloPoseDataset(Dataset):
    def __init__(self, img_dir, label_dir, img_size=640, nkpts=4, transform=None):
        self.img_files = sorted(glob.glob(os.path.join(img_dir, "*.jpg")) + glob.glob(os.path.join(img_dir, "*.png")))
        self.label_dir = label_dir
        self.img_size = img_size
        self.nkpts = nkpts
        self.transform = transform

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        img_path = self.img_files[index]
        label_path = os.path.join(self.label_dir, os.path.basename(img_path).rsplit('.', 1)[0] + ".txt")
        
        # 1. Image Load
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")
        
        img_resized = cv2.resize(img, (self.img_size, self.img_size))
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0

        # 2. Label Load
        targets = np.zeros((0, 6 + self.nkpts * 3))
        
        if os.path.exists(label_path):
            try:
                labels = np.loadtxt(label_path, ndmin=2)

                if labels.shape[0] > 0:
                    # Keypoint 처리
                    kpts = labels[:, 5:]
                    if kpts.shape[1] // self.nkpts == 2:
                        kpts = kpts.reshape(-1, self.nkpts, 2)
                        vis = np.ones((len(labels), self.nkpts, 1))
                        kpts = np.concatenate([kpts, vis], axis=2).reshape(-1, self.nkpts * 3)
                    boxes = labels[:, 1:5].copy()
                    cls = labels[:, 0:1].copy()
                    zeros = np.zeros((len(labels), 1))
                    
                    targets = np.concatenate([zeros, cls, boxes, kpts], axis=1)
            except Exception as e:
                print(f"Error loading label {label_path}: {e}")
                targets = np.zeros((0, 6 + self.nkpts * 3))

        labels_out = torch.from_numpy(targets).float()
        return img_tensor, labels_out
        
    @staticmethod
    def collate_fn(batch):
        imgs, labels = zip(*batch)
        imgs = torch.stack(imgs, 0)
        new_labels = []
        for i, label in enumerate(labels):
            if label.shape[0] > 0:
                l = label.clone()
                l[:, 0] = i
                new_labels.append(l)
        labels = torch.cat(new_labels, 0) if new_labels else torch.zeros((0, 6 + 4*3))
        return imgs, labels
