import os, sys, random
import numpy as np
import atexit
import signal
import torch
import torch.distributed as dist
from datetime import timedelta
from utils import line, printM, printS, printE, printW, MASTER_RANK

def set_seed(seed, use_deterministic=True):
    """
    재현성을 위한 시드 설정
    
    Args:
        seed: 랜덤 시드 값
        use_deterministic: True면 완전 결정론적(느림), False면 빠르지만 약간의 비결정성
    """
    # 환경변수 설정
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Python/NumPy/PyTorch 시드
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 멀티 GPU
    
    if use_deterministic:
        # 완전한 재현성 (느림)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)  # warn_only 추가
    else:
        # 빠른 학습 (약간의 비결정성)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


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
        self.ddp_initialized = False
        self.setup_device()

    @line
    def setup_device(self):
        printM(" [GPU Setting]\n", 'blue')
        if self.cpu or not torch.cuda.is_available():
            if not torch.cuda.is_available():
                printW('CUDA is not available')
            self.set_cpu()
        elif self.num_gpus == 1:
            self.set_cuda()
        else:
            if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
                self.set_ddp()
            else:
                printE(f" [Warning] {self.num_gpus}개의 GPU를 요청했으나 DDP 환경이 감지되지 않았습니다.")
                printM(" 싱글 GPU 모드로 전환합니다.\n", 'blue')
                self.set_cuda()

    def optimize_cpu_threads(self, num_gpus):
        """CPU 스레드 최적화"""
        cpu_count = os.cpu_count()
        optimal_threads = max(1, cpu_count // num_gpus // 2)
        os.environ['OMP_NUM_THREADS'] = str(optimal_threads)
        printS(f"OMP_NUM_THREADS set to {optimal_threads}")
        
    def set_ddp(self):
        """DDP 설정 (안전 장치 포함)"""
        
        # 환경변수 검증
        if 'LOCAL_RANK' not in os.environ:
            raise RuntimeError(
                " DDP 모드는 torchrun으로 실행해야 합니다.\n"
                " 예: torchrun --nproc_per_node=2 train.py"
            )
        
        # DDP 초기화
        try:
            dist.init_process_group(backend='nccl', timeout=timedelta(seconds=30))

        except Exception as e:
            raise RuntimeError(f" DDP 초기화 실패: {e}")
        
        local_rank_idx = int(os.environ['LOCAL_RANK'])
        
        # GPU 매핑 검증
        if not self.target_gpu_ids:
            dist.destroy_process_group()
            raise ValueError(" target_gpu_ids가 비어있습니다!")
        
        if local_rank_idx >= len(self.target_gpu_ids):
            dist.destroy_process_group()
            raise IndexError(
                printS(f"프로세스 수({local_rank_idx + 1})가 GPU 개수({len(self.target_gpu_ids)})보다 많습니다!")
            )
        
        real_gpu_id = self.target_gpu_ids[local_rank_idx]
        
        # GPU 존재 확인
        available_gpus = torch.cuda.device_count()
        
        if available_gpus == 0:
            dist.destroy_process_group()
            raise RuntimeError("CUDA GPU를 찾을 수 없습니다!")        
        try:
            assert printE(real_gpu_id >= available_gpus, f"GPU{self.target_gpu_ids}가 존재하지 않습니다. 사용가능한 GPU를 확인해주세요.")
        except:
            dist.barrier()
            dist.destroy_process_group()
            self.set_cuda()
            return
        
        # 디바이스 설정
        torch.cuda.set_device(real_gpu_id)
        self.device = torch.device(f'cuda:{real_gpu_id}')
        
        # 분산 정보 저장
        self.rank = dist.get_rank()
        self.local_rank = local_rank_idx
        self.world_size = dist.get_world_size()
        
        # 정보 출력 (마스터만)
        printS(f"DDP Mode Activated")
        printS(f"World Size: {self.world_size}")
        printS(f"Backend: nccl")
        printS(f"Target GPUs: {self.target_gpu_ids}")
        
        # 각 프로세스 매핑 출력
        printS(f" [Rank {self.rank:2d}] Local Rank {local_rank_idx} -> Physical GPU {real_gpu_id}")
        self.optimize_cpu_threads(available_gpus)
        # 동기화 (모든 프로세스가 여기까지 도달할 때까지 대기)
        dist.barrier()
        self.ddp_initialized = True
        
        # 종료 시그널 처리
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        atexit.register(self.cleanup)
                
    def _signal_handler(self, signum, frame):
        printS(f"\n\n Received signal {signum}, cleaning up...")
        self.cleanup()
        sys.exit(0)
    
    def cleanup(self):
        """DDP 정리"""
        if dist.is_initialized():
            if self.ddp_initialized:
                dist.barrier()  # 모든 프로세스 동기화
            dist.destroy_process_group()

    def set_cuda(self):
        """Single GPU 모드 설정"""
        # Single GPU 모드 기본값
        self.rank = 0
        self.world_size = 1
        self.local_rank = 0  # Single GPU는 항상 0
        
        # target_gpu_ids 유효성 검사
        if not self.target_gpu_ids:
            printW("target_gpu_ids가 비어있습니다. GPU 0번을 사용합니다.")
            gpu_id = 0
        else:
            gpu_id = self.target_gpu_ids[0]
        
        # GPU 존재 여부 확인
        available_gpus = torch.cuda.device_count()
        if available_gpus == 0:
            raise RuntimeError(" CUDA 사용 가능한 GPU가 없습니다!")
        
        if gpu_id >= available_gpus:
            printW(f"GPU {gpu_id}가 존재하지 않습니다. (사용 가능: 0-{available_gpus-1})")
            gpu_id = 0
        
        self.optimize_cpu_threads(available_gpus)
        self.device = torch.device(f'cuda:{gpu_id}')
        printS(f"Using Single GPU : {gpu_id}")

    def set_cpu(self):
        self.rank = 0
        self.world_size = 1
        self.local_rank = 0
        self.device = torch.device('cpu')
        printS("Using CPU")
