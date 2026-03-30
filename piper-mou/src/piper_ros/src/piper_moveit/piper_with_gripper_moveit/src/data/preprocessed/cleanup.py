#!/usr/bin/env python3
# cleanup.py
#
# 역할:
# - 이 파일의 워킹 디렉토리 기준으로 하위 폴더를 재귀적으로 순회
# - 확장자가 .npz, .json 인 파일을 모두 삭제
#
# 주의:
# - ./train/runs/lstm/ 경로 하위에 있는 파일/폴더는 절대 건드리지 않음

from pathlib import Path

# =========================
# 설정
# =========================
ROOT_DIR = Path.cwd()
PROTECT_DIR = (ROOT_DIR / "train" / "runs" / "lstm").resolve()
TARGET_EXTS = {".npz", ".json"}

# =========================
# 유틸
# =========================
def is_under_protected(path: Path) -> bool:
    """
    path가 보호 디렉토리(PROTECT_DIR) 하위에 있으면 True
    """
    try:
        path.resolve().relative_to(PROTECT_DIR)
        return True
    except ValueError:
        return False

# =========================
# main
# =========================
def main():
    removed = 0
    skipped = 0

    for p in ROOT_DIR.rglob("*"):
        if not p.is_file():
            continue

        if p.suffix not in TARGET_EXTS:
            continue

        if is_under_protected(p):
            skipped += 1
            continue

        try:
            p.unlink()
            print(f"[DEL] {p}")
            removed += 1
        except Exception as e:
            print(f"[FAIL] {p}: {e}")

    print(
        f"[DONE] removed={removed}, skipped(protected)={skipped}, "
        f"exts={sorted(TARGET_EXTS)}, protect={PROTECT_DIR}"
    )

if __name__ == "__main__":
    main()