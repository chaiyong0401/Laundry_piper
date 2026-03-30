# train_val_test_split.py
# data/preprocessed/에 *.npz로 저장되어 있는 파일들을 test/, train/, val/ 하위 디렉토리에 각각 split하여 *.npz 형태로 저장
# + split_manifest.json 저장

from __future__ import annotations
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

# ============================================================
# ✅ 설정 (여기만 수정)
# ============================================================
BASE_DIR = Path(__file__).parent
IN_DIR = BASE_DIR
SPLIT_DIRS = {
    "train": BASE_DIR / "train",
    "val": BASE_DIR/ "val",
    "test": BASE_DIR / "test",
}

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

SEED = 42
OVERWRITE = False # False면 기존 파일 있으면 스킵
MOVE_FILES = False # True면 move, False면 copy
STRATIFY_BY_SCENARIO = True # 가능한 범위에서 scenario 분포 유지 시도

MANIFEST_PATH = BASE_DIR / "split_manifest.json"

# ============================================================
# 유틸
# ============================================================
def _episode_files(in_dir: Path) -> List[Path]:
    # Train/ val/ test 폴더 제외하고, preprocessed 바로 아래 *.npz만
    out = []
    for fp in sorted(in_dir.glob("*.npz")):
        if fp.parent == in_dir:
            out.append(fp)
    return out

def _read_meta(npz_path: Path) -> Tuple[int, str]:
    with np.load(npz_path, allow_pickle=True) as d:
        scenario = int(d["scenario"]) if "scenario" in d else -1
        tag = str(d["tag"]) if "tag" in d else ""
    return scenario, tag
    
def _ensure_dirs():
    for p in SPLIT_DIRS.values():
        p.mkdir(parents=True, exist_ok = True)
        
def _split_counts(n:int) -> Tuple[int, int, int]:
    n_train = int(round(n * TRAIN_RATIO))
    n_val = int(round(n * VAL_RATIO))
    n_test = n - n_train - n_val
    if n_test < 0:
        raise ValueError("n_test is less than 0.")
    return n_train, n_val, n_test

def _assign_random(files: List[Path], rng: np.random.Generator) -> Dict[str, List[Path]]:
    files = files.copy()
    rng.shuffle(files)
    n_train, n_val, n_test = _split_counts(len(files))
    return {
        "train":  files[:n_train],
        "val": files[n_train:n_train+n_val],
        "test": files[n_train + n_val: n_train + n_val + n_test],
    }
    
def _assign_stratified_by_scenario(files: List[Path], rng:np.random.Generator) -> Dict[str, List[Path]]:
    '''
    각 scenario 안에서 train/val/test 비율로 먼저 나눈 다음, 그 결과를 합쳐서 전체 split에도 scenario가 고르게 섞이게 한다. 
    '''
    buckets: Dict[int, List[Path]] = {} # key가 Int고 item이 List[Path]임을 알려줌
    for fp in files:
        scenario, _ = _read_meta(fp)
        buckets.setdefault(scenario, []).append(fp) # buckets에 scenario라는 키가 있으면 그 리스트에 fp를 추가, 없으면 빈 리스트 [] 만들어서 그 리스트에 fp 추가
        
    out = {"train": [], "val": [], "test":[]}
    for sc, fps in buckets.items():
        fps = fps.copy()
        rng.shuffle(fps)
        n_train, n_val, n_test = _split_counts(len(fps))
        out["train"].extend(fps[:n_train])
        out["val"].extend(fps[n_train:n_train + n_val])
        out["test"].extend(fps[n_train + n_val: n_train + n_val + n_test])
        
    for k in out:
        rng.shuffle(out[k])
    return out
    
def _copy_or_move(src: Path, dst: Path, move: bool):
    dst.parent.mkdir(parents = True, exist_ok = True)
    if move:
        shutil.move(str(src), str(dst))
    else:
        shutil.copy2(str(src), str(dst))
        

# ============================================================
# main
# ============================================================
def main():
    rng = np.random.default_rng(SEED)
    _ensure_dirs()
    
    episodes = _episode_files(IN_DIR)
    if not episodes:
        print(f"[WARN] No episode npz files in {IN_DIR}")
        return
    
    # split 대상에서 이미 split 폴더로 들어간 애들 제외(안전장치)
    episodes = [fp for fp in episodes if fp.parent == IN_DIR]
    
    print(f"[INFO] Found {len(episodes)} episodes in {IN_DIR}")
    
    if abs((TRAIN_RATIO + VAL_RATIO + TEST_RATIO) - 1.0) > 1e-6:
        raise ValueError("TRAIN_RATIO + VAL_RATIO + TEST_RATIO must be 1.0")
    
    if STRATIFY_BY_SCENARIO:
        assignment = _assign_stratified_by_scenario(episodes, rng)
        print("[INFO] Split mode: stratified by scenario")
    else:
        assignment = _assign_random(episodes, rng)
        print("[INFO] Split mode: random")
    
    manifest = {
        "config": {
            "train_ratio": TRAIN_RATIO,
            "val_ratio": VAL_RATIO,
            "test_ratio": TEST_RATIO,
            "seed": SEED,
            "move_files": MOVE_FILES,
            "stratify_by_scenario": STRATIFY_BY_SCENARIO,
        },
        "items": []
    }
    
    for split, files in assignment.items():
        out_dir = SPLIT_DIRS[split]
        print(f"[INFO] {split}: {len(files)} episodes -> {out_dir}")
        
        for src in files:
            dst = out_dir / src.name
            if dst.exists() and not OVERWRITE:
                print(f"[SKIP] {split}/{src.name} (exists)")
                scenario, tag = _read_meta(dst)
            else:
                scenario, tag = _read_meta(src)
                _copy_or_move(src, dst, move=MOVE_FILES)
                print(f"[OK] {src.name} -> {split}/{src.name}")
            
            manifest["items"].append({
                "episode": src.name,
                "split": split,
                "scenario": int(scenario),
                "tag": str(tag),
            })
    
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[DONE] Wrote manifest: {MANIFEST_PATH}")
    
    def _count(items, split):
        return sum(1 for x in items if x["split"] == split)
    
    print(f"[SUMMARY] train={_count(manifest['items'], 'train')}, "
          f"val={_count(manifest['items'], 'val')}, "
          f"test={_count(manifest['items'], 'test')}")

if __name__ == "__main__":
    main()