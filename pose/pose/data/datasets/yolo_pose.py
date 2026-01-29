import torch 
import os, glob, cv2
import numpy as np 
from torch.utils.data import Dataset
from tqdm import tqdm
from pose.utils import printS, printM, colored_msg

class YoloPoseDataset(Dataset):
    def __init__(self, img_dir, label_dir, img_size=640, nkpts=4, transform=None):
        self.img_files = sorted(glob.glob(os.path.join(img_dir, "*.jpg")) + glob.glob(os.path.join(img_dir, "*.png")))
        self.label_dir = label_dir
        self.img_size = img_size
        self.nkpts = nkpts
        self.transform = transform
        
        # [핵심] RAM 캐시 저장소 초기화
        # self.ram_cache = {} 
        
        # 라벨 미리 읽기 (기존 최적화 유지)
        printS(f"Loading labels for {len(self.img_files)} images...")
        self.cached_labels = []
        for img_path in tqdm(self.img_files, desc=f" {colored_msg('[SYSTEMS]', 'blue')} Parsing Labels"):
            label_path = os.path.join(self.label_dir, os.path.basename(img_path).rsplit('.', 1)[0] + ".txt")
            targets = np.zeros((0, 6 + self.nkpts * 3))
            
            if os.path.exists(label_path) and os.path.getsize(label_path) > 0:
                try:
                    labels = np.loadtxt(label_path, ndmin=2)
                    if labels.shape[0] > 0:
                        kpts = labels[:, 5:]
                        if kpts.shape[1] // self.nkpts == 2:
                            kpts = kpts.reshape(-1, self.nkpts, 2)
                            vis = np.ones((len(labels), self.nkpts, 1))
                            kpts = np.concatenate([kpts, vis], axis=2).reshape(-1, self.nkpts * 3)
                        boxes = labels[:, 1:5].copy()
                        cls = labels[:, 0:1].copy()
                        zeros = np.zeros((len(labels), 1))
                        targets = np.concatenate([zeros, cls, boxes, kpts], axis=1)
                except:
                    pass
            self.cached_labels.append(targets)

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        # if index in self.ram_cache:
        #     return self.ram_cache[index]

        img_path = self.img_files[index]
        
        # HDD 읽기
        img = cv2.imread(img_path)
        if img is None:
            img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        
        # Resize (1920 -> 640) CPU 연산
        if img.shape[0] != self.img_size or img.shape[1] != self.img_size:
            img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)

        # HWC -> CHW 변환
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        
        # Uint8 상태로 Tensor 변환 (Float 변환 및 나누기 금지 -> 메모리 절약)
        img_tensor = torch.from_numpy(img) 

        # 라벨 가져오기
        targets = self.cached_labels[index]
        labels_out = torch.from_numpy(targets).float()
        result = (img_tensor, labels_out)
        # self.ram_cache[index] = result
        
        return result
        
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