import os, random
import numpy as np
import torch
import torch.distributed as dist
from utils import line

def set_seed(seed):
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"
    os.environ['PYTHONHASHSEED']=str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)

class DDPManager:
    def __init__(self, gpus):
        self.cpu = gpus[0] == -1
        self.num_gpus = len(gpus)   # 2개
        self.target_gpu_ids = gpus  # 예: [2, 4]
        self.rank = 0
        self.local_rank = 0
        self.world_size = 1
        self.device = None
        self.is_master = True # 로그 출력 여부 등을 결정 (p)        
        self.setup_device()

    @line
    def setup_device(self):
        if self.cpu or not torch.cuda.is_available():
            self.set_cpu()
        elif self.num_gpus == 1:
            self.set_cuda()
        else:
            if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
                self.set_ddp()
            else:
                print(f"[Warning] {self.num_gpus}개의 GPU를 요청했으나 DDP 환경이 감지되지 않았습니다.")
                print("싱글 GPU 모드로 전환합니다.\n")
                self.set_cuda()

    @line
    def set_ddp(self):
        dist.init_process_group(backend='nccl')
        
        # 1. torchrun이 주는 가상 번호 (0, 1, 2...)
        local_rank_idx = int(os.environ['LOCAL_RANK']) 
        
        # 2. [중요] 가상 번호를 실제 내가 원하는 GPU 번호로 변환(Mapping)
        # local_rank가 0이면 -> real_gpu_id는 2
        # local_rank가 1이면 -> real_gpu_id는 4
        real_gpu_id = self.target_gpu_ids[local_rank_idx]
        
        # 3. 변환된 실제 번호로 세팅
        torch.cuda.set_device(real_gpu_id)
        self.device = torch.device(f'cuda:{real_gpu_id}')
        
        # 멤버 변수 저장
        self.rank = dist.get_rank()
        self.local_rank = local_rank_idx # 로직상 순번은 그대로 0, 1 유지
        self.world_size = dist.get_world_size()
        
        # 마스터 프로세스만 출력
        if self.rank == 0:
            print(f" DDP Activated")
            print(f"사용할 GPU 리스트: {self.target_gpu_ids}")
        
        # 각 프로세스가 자신이 사용하는 실제 GPU 번호 출력해보기
        print(f"[Rank {self.rank}] 논리 번호: {local_rank_idx} -> 물리 GPU: {real_gpu_id} 연결 완료")

    def set_cuda(self):
        self.rank = 0
        self.world_size = 1
        self.local_rank = 0
        self.device = torch.device('cuda:0')
        self.is_master = True
        print(" Single GPU Mode Activated")

    def set_cuda(self):
        self.rank = 0          # 혼자니까 무조건 0
        self.world_size = 1    # 혼자니까 무조건 1
        # target_gpu_ids가 실제로 존재하는지 확인
        if self.target_gpu_ids[0] in range(torch.cuda.device_count()):
            self.local_rank = self.target_gpu_ids[0]
        else:
            print(f" [Warning] 요청한 GPU ID {self.target_gpu_ids[0]}가 존재하지 않습니다. GPU 0번을 사용합니다.")
            self.local_rank = 0
        self.device = torch.device(f'cuda:{self.local_rank}')
        print(f" Using GPU: {self.local_rank}\n")
        print(" Single GPU Mode Activated")

    def set_cpu(self):
        self.rank = 0
        self.world_size = 1
        self.local_rank = 0
        self.device = torch.device('cpu')
        self.is_master = True
        print(" CPU Mode Activated")
