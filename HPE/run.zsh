#!/bin/bash

# 1. 입력 인자 확인
if [ "$#" -eq 0 ]; then
    echo "Usage: $0 <gpu_id1> <gpu_id2> ..."
    exit 1
fi

# 2. CPU 코어 수 확인
NUM_CORES=$(nproc)

# 3. GPU 개수 계산 (가장 중요!)
# [!] ${#GPU_IDS[@]} 대신 $#를 써야 정확합니다.
# $@를 변수에 넣으면 문자열 하나로 취급되어 개수가 1로 잘못 계산될 수 있습니다.
NUM_GPUS=$#

# 4. GPU 리스트 문자열 생성 (0,1,2,3 형태)
# 입력 인자 사이의 공백을 콤마로 변경
export GPU_LIST=$(echo "$@" | tr ' ' ',')

echo "--------------------------------"
echo "Detected CPU Cores: $NUM_CORES"
echo "Target GPUs ($NUM_GPUS): $GPU_LIST"
echo "--------------------------------"

# 5. 스레드 수 계산
if [ $NUM_GPUS -gt 0 ]; then
    OMP_THREADS=$((NUM_CORES / NUM_GPUS / 2))
else
    OMP_THREADS=1
fi
# 최소 1 보장
[ $OMP_THREADS -lt 1 ] && OMP_THREADS=1

echo "Setting OMP_NUM_THREADS=$OMP_THREADS"
export OMP_NUM_THREADS=$OMP_THREADS

# 6. 학습 시작
# [!] main.py에 --gpus 인자를 넘길 필요가 없는 경우가 많습니다 (torchrun 사용 시).
# 하지만 코드에서 필요로 한다면 그대로 둡니다.
torchrun --nproc_per_node=$NUM_GPUS main.py --gpus "$GPU_LIST"