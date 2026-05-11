import argparse
import shutil
from pathlib import Path
from datetime import datetime
import sys

# save_submission.py
# GitHub Copilot
# 使用方式範例:
#   python save_submission.py                      -> 使用預設路徑，upload_result.txt 內容為 0
#   python save_submission.py --upload 1           -> upload_result.txt 內容為 1
#   python save_submission.py --codeName "my baseline.py" --submission "out/sub.csv" --history "./history_dir" --upload 2


def make_unique_dir(base: Path, name: str) -> Path:
    candidate = base / name
    if not candidate.exists():
        candidate.mkdir(parents=True, exist_ok=False)
        return candidate
    # add suffix
    i = 1
    while True:
        c = base / f"{name}_{i}"
        if not c.exists():
            c.mkdir(parents=True, exist_ok=False)
            return c
        i += 1

def main():
    p = argparse.ArgumentParser(description="備份 baseline code.py 與 result/submission.csv 到 history 並建立 upload_result.txt")
    p.add_argument("--codeName", "-c", default="baseline code.py", help="baseline 檔案路徑 (預設: 'baseline code.py')")
    p.add_argument("--submission", "-s", default="result/submission.csv", help="submission 檔案路徑 (預設: 'result/submission.csv')")
    p.add_argument("--history", "-H", default="history", help="history 資料夾 (預設: 'history')")
    p.add_argument("--upload", "-u", default="0", help="寫入 upload_result.txt 的數字 (預設: 0)")
    args = p.parse_args()

    cwd = Path.cwd()
    baseline_path = (cwd / args.codeName).resolve()
    submission_path = (cwd / args.submission).resolve()
    history_base = (cwd / args.history).resolve()
    upload_value = str(args.upload)

    try:
        history_base.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"無法建立 history 資料夾: {e}", file=sys.stderr)
        sys.exit(2)

    # 建立一個名為 "backup" 的子資料夾，若已存在則用 _1, _2, ... 編號
    try:
        dest_dir = make_unique_dir(history_base, "backup")
    except Exception as e:
        print(f"無法建立備份資料夾: {e}", file=sys.stderr)
        sys.exit(3)

    copied = []
    for src in (baseline_path, submission_path):
        if src.exists():
            try:
                dest = dest_dir / src.name
                shutil.copy2(src, dest)
                copied.append(dest)
            except Exception as e:
                print(f"複製 {src} 失敗: {e}", file=sys.stderr)
        else:
            print(f"注意: 檔案不存在，跳過: {src}", file=sys.stderr)

    # 建立 upload_result.txt 並寫入數字
    try:
        upload_file = dest_dir / "upload_result.txt"
        upload_file.write_text(upload_value, encoding="utf-8")
    except Exception as e:
        print(f"寫入 upload_result.txt 失敗: {e}", file=sys.stderr)
        sys.exit(4)

    # 簡短輸出結果
    print(str(dest_dir))
    submission_path = (cwd / args.submission).resolve()
    history_base = (cwd / args.history).resolve()
    upload_value = str(args.upload)

    try:
        history_base.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"無法建立 history 資料夾: {e}", file=sys.stderr)
        sys.exit(2)

    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # try:
    #     dest_dir = make_unique_dir(history_base, timestamp)
    # except Exception as e:
    #     print(f"無法建立備份資料夾: {e}", file=sys.stderr)
    #     sys.exit(3)

    copied = []
    for src in (baseline_path, submission_path):
        if src.exists():
            try:
                dest = dest_dir / src.name
                shutil.copy2(src, dest)
                copied.append(dest)
            except Exception as e:
                print(f"複製 {src} 失敗: {e}", file=sys.stderr)
        else:
            print(f"注意: 檔案不存在，跳過: {src}", file=sys.stderr)

    # 建立 upload_result.txt 並寫入數字
    try:
        upload_file = dest_dir / "upload_result.txt"
        upload_file.write_text(upload_value, encoding="utf-8")
    except Exception as e:
        print(f"寫入 upload_result.txt 失敗: {e}", file=sys.stderr)
        sys.exit(4)

    # 簡短輸出結果
    print(str(dest_dir))

if __name__ == "__main__":
    main()