#!/bin/bash

# 1. 입력 인자 확인
if [ "$#" -eq 0 ]; then
    echo "Usage: $0 <gpu_id1> <gpu_id2> ... [config_file]"
    echo "  gpu_ids: GPU IDs to use (e.g., 0 1 2 3)"
    echo "  config_file: Configuration file path (optional, default: configs/few_shot/fskd_small.yaml)"
    exit 1
fi

# 2. 파라미터 파싱
# 마지막 인자가 파일 경로 패턴(.yaml, .yml, /configs/)이면 config file로 인식
LAST_ARG="${!#}"
if [[ "$LAST_ARG" == *.yaml || "$LAST_ARG" == *.yml || "$LAST_ARG" == *"/configs/"* ]]; then
    # 마지막 인자가 config 파일인 경우
    GPU_IDS=("${@:1:$#-1}")
    CFG_FILE="$LAST_ARG"
else
    # 마지막 인자가 숫자이면 모든 인자를 GPU ID로 간주, config는 기본값 사용
    GPU_IDS=("$@")
    CFG_FILE="configs/method/fskd_small.yaml"
fi

# GPU_IDS 검증 (숫자만 포함되어야 함)
VALID_GPU_IDS=()
for gpu_id in "${GPU_IDS[@]}"; do
    if [[ "$gpu_id" =~ ^[0-9]+$ ]]; then
        VALID_GPU_IDS+=("$gpu_id")
    fi
done

# 검증된 GPU_ID로 업데이트
if [ ${#VALID_GPU_IDS[@]} -gt 0 ]; then
    GPU_IDS=("${VALID_GPU_IDS[@]}")
else
    echo "Error: No valid GPU IDs found"
    exit 1
fi

# 3. CPU 코어 수 확인
NUM_CORES=$(nproc)

# 4. GPU 개수 계산
NUM_GPUS=${#GPU_IDS[@]}

# 5. GPU 리스트 문자열 생성 (0,1,2,3 형태)
export GPU_LIST=$(printf "%s," "${GPU_IDS[@]}")
# 마지막 콤마 제거
export GPU_LIST=${GPU_LIST%,}

echo "--------------------------------"
echo "Detected CPU Cores: $NUM_CORES"
echo "Target GPUs ($NUM_GPUS): $GPU_LIST"
echo "Configuration File: $CFG_FILE"
echo "--------------------------------"

# 6. 스레드 수 계산
if [ $NUM_GPUS -gt 0 ]; then
    OMP_THREADS=$((NUM_CORES / NUM_GPUS / 2))
else
    OMP_THREADS=1
fi
# 최소 1 보장
[ $OMP_THREADS -lt 1 ] && OMP_THREADS=1

echo "Setting OMP_NUM_THREADS=$OMP_THREADS"
export OMP_NUM_THREADS=$OMP_THREADS

# 7. Configuration file 존재 확인
if [ ! -f "$CFG_FILE" ]; then
    echo "Error: Configuration file not found: $CFG_FILE"
    exit 1
fi

# 8. 학습 시작
echo "Starting training with configuration: $CFG_FILE"
torchrun --nproc_per_node=$NUM_GPUS main.py --gpus "$GPU_LIST" --cfg "$CFG_FILE"
