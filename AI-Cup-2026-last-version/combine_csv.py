import argparse
import sys

import pandas as pd

DESIRED_COLS = ["rally_uid", "actionId", "pointId", "serverGetPoint"]


def load_task_column(path: str, col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "rally_uid" not in df.columns:
        raise ValueError(f"{path} 缺少 rally_uid 欄位")
    if col not in df.columns:
        raise ValueError(f"{path} 缺少 {col} 欄位")
    dup = int(df["rally_uid"].duplicated().sum())
    if dup:
        raise ValueError(f"{path} 有 {dup} 個重複的 rally_uid，請先處理")
    return df[["rally_uid", col]]


def rally_uid_set(path: str) -> set:
    df = pd.read_csv(path)
    if "rally_uid" not in df.columns:
        raise ValueError(f"{path} 缺少 rally_uid 欄位")
    dup = int(df["rally_uid"].duplicated().sum())
    if dup:
        raise ValueError(f"{path} 有 {dup} 個重複的 rally_uid，請先處理")
    return set(df["rally_uid"])


def _format_uid_sample(uids: set, limit: int = 5) -> str:
    sample = sorted(uids)[:limit]
    suffix = f" ...（共 {len(uids)} 筆）" if len(uids) > limit else ""
    return f"{sample}{suffix}"


def require_matching_rally_uids(sources: dict[str, str]) -> set:
    """三份來源的 rally_uid 必須完全一致，否則中止。"""
    uid_sets = {name: rally_uid_set(path) for name, path in sources.items()}
    names = list(uid_sets.keys())
    ref_name, ref_set = names[0], uid_sets[names[0]]
    errors = []

    for name in names[1:]:
        other = uid_sets[name]
        if other == ref_set:
            continue
        only_ref = ref_set - other
        only_other = other - ref_set
        errors.append(
            f"  [{ref_name}] vs [{name}]："
            f"{ref_name} {len(ref_set)} 筆，{name} {len(other)} 筆"
        )
        if only_ref:
            errors.append(f"    僅在 {ref_name}: {_format_uid_sample(only_ref)}")
        if only_other:
            errors.append(f"    僅在 {name}: {_format_uid_sample(only_other)}")

    if errors:
        print("❌ 三份來源的 rally_uid 不一致，已中止合併：", file=sys.stderr)
        for line in errors:
            print(line, file=sys.stderr)
        for name, path in sources.items():
            print(f"     {name}: {path}", file=sys.stderr)
        sys.exit(1)

    print(f"✅ 三份來源 rally_uid 完全一致（{len(ref_set)} 筆）")
    return ref_set


def main(args: argparse.Namespace) -> None:
    rally_src = args.rally if args.rally else args.action
    sources = {
        "action": args.action,
        "point": args.point,
        "rally": rally_src,
    }

    require_matching_rally_uids(sources)

    act_df = load_task_column(args.action, "actionId")
    pt_df = load_task_column(args.point, "pointId")
    rly_df = load_task_column(rally_src, "serverGetPoint")

    final_submission = (
        act_df.merge(pt_df, on="rally_uid", how="inner")
        .merge(rly_df, on="rally_uid", how="inner")
        .sort_values("rally_uid")
        .reset_index(drop=True)[DESIRED_COLS]
    )

    final_submission.to_csv(args.out, index=False)
    print("科學怪人縫合完畢！")
    print(f"  actionId       <- {args.action}")
    print(f"  pointId        <- {args.point}")
    print(f"  serverGetPoint <- {rally_src}")
    print(f"  輸出 -> {args.out} ({len(final_submission)} 筆)")
    print(f"  欄位順序: {final_submission.columns.tolist()}")
    print("準備上傳！")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="從不同 CSV 合併 actionId / pointId / serverGetPoint 成一份 submission"
    )
    ap.add_argument(
        "--action",
        default="output/baseline_bilstm2_act_pt_rly.csv",
        help="actionId 來源 CSV（需含 rally_uid, actionId）",
    )
    ap.add_argument(
        "--point",
        default="output/train_point_only.csv",
        help="pointId 來源 CSV（需含 rally_uid, pointId）",
    )
    ap.add_argument(
        "--rally",
        default=None,
        help="serverGetPoint 來源 CSV；未指定時與 --action 相同",
    )
    ap.add_argument(
        "--out",
        default="output/BaseMod3_TPO_2.csv",
        help="合併後輸出路徑",
    )
    main(ap.parse_args())
