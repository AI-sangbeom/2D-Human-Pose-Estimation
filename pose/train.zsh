#!/bin/bash

# 1. 입력 인자 확인
if [ "$#" -eq 0 ]; then
    echo "Usage: $0 [config_file] <gpu_id1> <gpu_id2> ..."
    echo "  config_file: Configuration file path (optional, default: configs/method/fskd_small.yaml)"
    echo "  gpu_ids: GPU IDs to use (e.g., 0 1 2 3)"
    exit 1
fi

# 2. 파라미터 파싱 (수정됨)
# 첫 번째 인자가 파일 경로 패턴(.yaml, .yml)이거나 실제 존재하는 파일이면 config file로 인식
FIRST_ARG="$1"

if [[ "$FIRST_ARG" == *.yaml || "$FIRST_ARG" == *.yml || -f "$FIRST_ARG" ]]; then
    # 첫 번째 인자가 config 파일인 경우
    CFG_FILE="$FIRST_ARG"
    # 첫 번째 인자를 제외한 나머지를 GPU ID로 설정 (${@:2}는 2번째부터 끝까지)
    GPU_IDS=("${@:2}")
else
    # 첫 번째 인자가 config 파일이 아니면(숫자 등), 기본 config 사용 및 모든 인자를 GPU ID로 간주
    GPU_IDS=("$@")
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
    # GPU ID가 하나도 없거나 유효하지 않은 경우 0번 할당
    GPU_IDS=(0)
    echo "Warning: No valid GPU IDs found in arguments."
    echo "Auto-assigning GPU ID 0"
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

export OMP_NUM_THREADS=$OMP_THREADS

# torchrun 실행
export TORCH_USE_CUDA_DSA=1
export CUDA_LAUNCH_BLOCKING=1
torchrun --nproc_per_node=$NUM_GPUS train.py --gpus "$GPU_LIST" --cfg "$CFG_FILE"