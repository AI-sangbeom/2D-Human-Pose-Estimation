from pathlib import Path
import natsort
# ================= 설정 영역 =================
# 레이블 파일이 있는 경로
LABEL_DIR = Path('/media/otter/otterHD/AXData/TotalAX/TotalAX/valid/labels')

# 이미지 파일이 있는 경로 (보통 labels와 같은 레벨의 images 폴더에 위치함)
# 만약 레이블과 이미지가 같은 폴더에 있다면 아래를 LABEL_DIR과 동일하게 설정하세요.
IMAGE_DIR = LABEL_DIR.parent / "images" 


NUM_KPTS = 4  # 키포인트 개수
EXPECTED_COLS = 5 + 2 * NUM_KPTS  # 예상되는 컬럼 수 (class + x + y + w + h + kpts...)

# True: 삭제하지 않고 삭제될 파일 목록만 출력 (안전 모드)
# False: 실제로 파일 삭제 수행
DRY_RUN = True 
# ===========================================

def find_image_path(label_path, image_dir):
    """레이블 파일명과 동일한 이미지 파일을 찾습니다."""
    # 체크할 이미지 확장자 목록
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    
    for ext in valid_extensions:
        img_path = image_dir / f"{label_path.stem}{ext}"
        if img_path.exists():
            return img_path
    return None

def main():
    print(f"Start scanning in: {LABEL_DIR}")
    print(f"Expected columns: {EXPECTED_COLS}")
    
    bad_files_count = 0
    deleted_files_count = 0

    # 레이블 폴더 내 모든 txt 파일 순회
    for label_file in natsort.natsorted(LABEL_DIR.glob("*.txt")):
        is_bad = False
        
        try:
            with open(label_file, 'r') as f:
                lines = f.readlines()
                
            # 빈 파일이거나, 컬럼 수가 맞지 않는 라인이 있는지 확인
            if not lines:
                is_bad = False # 빈 파일도 삭제 대상이라면 True
            else:
                for i, line in enumerate(lines):
                    cols = len(line.strip().split())
                    # 라인이 비어있지 않은데 컬럼 수가 안 맞으면 불량
                    if line.strip() and (cols != EXPECTED_COLS):
                        is_bad = True
                        print(f"[Bad Label detected] {label_file.name} (Line {i+1}: {cols} cols)")
                        break
        except Exception as e:
            print(f"Error reading {label_file}: {e}")
            continue

        # 불량 레이블로 판명되면 삭제 로직 수행
        if is_bad:
            bad_files_count += 1
            
            # 1. 이미지 파일 찾기
            image_file = find_image_path(label_file, IMAGE_DIR)
            
            # 2. 삭제 (DRY_RUN 체크)
            if DRY_RUN:
                print(f"  [DRY RUN] Would delete label: {label_file}")
                if image_file:
                    print(f"  [DRY RUN] Would delete image: {image_file}")
                else:
                    print(f"  [DRY RUN] Image not found for: {label_file.name}")
            else:
                # 실제 삭제 수행
                try:
                    label_file.unlink() # 레이블 삭제
                    print(f"  [DELETED] Label: {label_file.name}")
                    
                    if image_file:
                        image_file.unlink() # 이미지 삭제
                        print(f"  [DELETED] Image: {image_file.name}")
                    else:
                        print(f"  [WARNING] Image not found, only label deleted.")
                    
                    deleted_files_count += 1
                except Exception as e:
                    print(f"  [ERROR] Failed to delete files: {e}")

    print("-" * 30)
    if DRY_RUN:
        print(f"Scan complete. Found {bad_files_count} bad files. (No files were deleted).")
        print("Set DRY_RUN = False to execute deletion.")
    else:
        print(f"Process complete. Deleted {deleted_files_count} pairs out of {bad_files_count} bad labels found.")

if __name__ == "__main__":
    main()