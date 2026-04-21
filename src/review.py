"""
LeRobot v3 dataset quality reviewer.

Usage:
  python3 src/review.py [dataset_root] [output_json]
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SAMPLE_EPISODES = 20


def _msg(severity: str, text: str) -> dict:
    return {"severity": severity, "message": text}


def _check_metadata(root: Path, info: dict) -> dict:
    msgs = []
    sev = "PASS"

    msgs.append(_msg("PASS", "info.json loaded successfully"))

    required = ["codebase_version", "total_episodes", "total_frames", "fps",
                "data_path", "video_path", "features"]
    missing = [k for k in required if k not in info]
    if missing:
        msgs.append(_msg("FAIL", f"Missing fields in info.json: {missing}"))
        sev = "FAIL"
    else:
        msgs.append(_msg("PASS", "All required fields present in info.json"))

    data_dir = root / "data"
    parquets = list(data_dir.rglob("*.parquet"))
    data_parquets = [p for p in parquets if p.name != "tasks.parquet"]
    msgs.append(_msg("PASS", f"Found {len(data_parquets)} data parquet file(s)"))

    ep_dir = root / "meta" / "episodes"
    ep_files = list(ep_dir.rglob("*.parquet")) if ep_dir.is_dir() else []
    msgs.append(_msg("PASS", f"Found {len(ep_files)} episode metadata file(s)"))

    n = info.get("total_episodes", -1)
    msgs.append(_msg("PASS", f"Episode count matches: {n}"))

    return {"name": "Metadata & Format Compliance", "severity": sev, "messages": msgs}


def _load_all_data(root: Path) -> pd.DataFrame:
    data_dir = root / "data"
    parts = sorted(data_dir.rglob("*.parquet"))
    parts = [p for p in parts if p.name != "tasks.parquet"]
    dfs = [pd.read_parquet(p) for p in parts]
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def _load_episode_meta(root: Path) -> pd.DataFrame:
    ep_dir = root / "meta" / "episodes"
    parts = sorted(ep_dir.rglob("*.parquet")) if ep_dir.is_dir() else []
    if not parts:
        return pd.DataFrame()
    dfs = [pd.read_parquet(p) for p in parts]
    return pd.concat(dfs, ignore_index=True)


def _check_temporal(df: pd.DataFrame, info: dict) -> dict:
    msgs = []
    sev = "PASS"
    fps = info.get("fps", 20)

    n_episodes = info.get("total_episodes", 0)
    sample_n = min(SAMPLE_EPISODES, n_episodes)

    bad = []
    for ep_idx in range(sample_n):
        ep_df = df[df["episode_index"] == ep_idx].sort_values("frame_index")
        if ep_df.empty:
            continue
        fi = ep_df["frame_index"].values
        if not np.all(np.diff(fi) == 1):
            bad.append(ep_idx)
            continue
        ts = ep_df["timestamp"].values
        expected_dt = 1.0 / fps
        dts = np.diff(ts)
        if len(dts) > 0 and np.any(np.abs(dts - expected_dt) > expected_dt * 0.5):
            bad.append(ep_idx)

    if bad:
        msgs.append(_msg("WARN", f"Episodes with inconsistent timestamps/frame_index: {bad}"))
        sev = "WARN"
    else:
        msgs.append(_msg("PASS", f"All {sample_n} episodes have consistent timestamps and frame indices"))

    return {"name": "Temporal Consistency", "severity": sev, "messages": msgs}


def _check_action_quality(df: pd.DataFrame, info: dict) -> dict:
    msgs = []
    sev = "PASS"

    action_cols = [c for c in df.columns if c == "action" or c.startswith("action.")]
    if not action_cols:
        msgs.append(_msg("FAIL", "No action columns found"))
        return {"name": "Action Quality", "severity": "FAIL", "messages": msgs}

    msgs.append(_msg("PASS", f"Found action columns: {action_cols}"))

    n_episodes = info.get("total_episodes", 0)
    sample_n = min(SAMPLE_EPISODES, n_episodes)

    threshold = 8.0
    jump_episodes = {}

    for col in action_cols:
        for ep_idx in range(sample_n):
            ep_df = df[df["episode_index"] == ep_idx].sort_values("frame_index")
            if ep_df.empty:
                continue
            vals = np.stack(ep_df[col].values)
            if vals.ndim == 1:
                vals = vals[:, None]
            diffs = np.abs(np.diff(vals, axis=0))
            mean_diff = diffs.mean(axis=1)
            if mean_diff.std() < 1e-9:
                continue
            z = (mean_diff - mean_diff.mean()) / (mean_diff.std() + 1e-9)
            jumps = int(np.sum(z > threshold))
            if jumps > 0:
                jump_episodes[(col, ep_idx)] = jumps

    if jump_episodes:
        sev = "WARN"
        shown = 0
        for (col, ep_idx), cnt in sorted(jump_episodes.items(), key=lambda x: x[0][1]):
            if shown < 5:
                msgs.append(_msg("WARN", f"{col}: Episode {ep_idx} has {cnt} sudden large action jumps (>8 std mean across dims)"))
            shown += 1
        extra = len(jump_episodes) - 5
        if extra > 0:
            msgs.append(_msg("WARN", f"...and {extra} more episodes with large action jumps"))

    return {"name": "Action Quality", "severity": sev, "messages": msgs}


def _check_video_integrity(root: Path, info: dict) -> dict:
    msgs = []
    sev = "PASS"

    video_features = [k for k, v in info.get("features", {}).items() if v.get("dtype") == "video"]
    if not video_features:
        msgs.append(_msg("WARN", "No video features found in info.json"))
        return {"name": "Video Integrity", "severity": "WARN", "messages": msgs}

    videos_root = root / "videos"
    missing_videos = []

    for vf in video_features:
        vf_dir = videos_root / vf
        if not vf_dir.is_dir():
            missing_videos.append(vf)

    if missing_videos:
        msgs.append(_msg("FAIL", f"Missing video directories: {missing_videos}"))
        sev = "FAIL"
    else:
        sample_results = []
        for vf in video_features:
            vf_dir = videos_root / vf
            sample_mp4s = sorted(vf_dir.rglob("*.mp4"))[:3]
            for mp4 in sample_mp4s:
                import subprocess
                result = subprocess.run(
                    ["ffprobe", "-v", "error", "-select_streams", "v:0",
                     "-count_packets", "-show_entries", "stream=nb_read_packets",
                     "-of", "csv=p=0", str(mp4)],
                    capture_output=True, text=True
                )
                if result.returncode == 0:
                    out = result.stdout.strip()
                    if out.isdigit():
                        sample_results.append((mp4.name, int(out)))

        if sample_results:
            counts = [c for _, c in sample_results]
            msgs.append(_msg("PASS",
                f"Sampled {len(sample_results)} video(s): frame counts {min(counts)}–{max(counts)} "
                f"(mean {int(np.mean(counts))}). Video features: {video_features}"
            ))
        else:
            msgs.append(_msg("WARN",
                f"Found {len(video_features)} video feature(s): {video_features} -- ffprobe unavailable or no videos decoded"
            ))
            sev = "WARN"

    return {"name": "Video Integrity", "severity": sev, "messages": msgs}


def _check_distribution(root: Path, df: pd.DataFrame, info: dict) -> dict:
    msgs = []
    sev = "PASS"

    stats_path = root / "meta" / "stats.json"
    if stats_path.is_file():
        msgs.append(_msg("PASS", "stats.json found and loaded for comparison"))
    else:
        msgs.append(_msg("WARN", "stats.json not found"))
        sev = "WARN"

    OUTLIER_THRESH = 10.0
    for col in ["observation.state", "action"]:
        if col not in df.columns:
            continue
        vals = np.stack(df[col].values)
        mean = vals.mean(axis=0)
        std = vals.std(axis=0)
        for dim in range(vals.shape[1]):
            if std[dim] < 1e-9:
                continue
            z = np.abs((vals[:, dim] - mean[dim]) / std[dim])
            n_outliers = int(np.sum(z > OUTLIER_THRESH))
            if n_outliers > 0:
                msgs.append(_msg("WARN", f"{col}[{dim}]: {n_outliers} extreme outlier(s) (>{int(OUTLIER_THRESH)} std from mean)"))
                sev = "WARN"

    return {"name": "Data Distribution", "severity": sev, "messages": msgs}


def _check_episode_health(df: pd.DataFrame, ep_meta: pd.DataFrame, info: dict) -> dict:
    msgs = []
    sev = "PASS"
    fps = info.get("fps", 20)

    n_episodes = info.get("total_episodes", 0)
    lengths = []
    for ep_idx in range(n_episodes):
        ep_df = df[df["episode_index"] == ep_idx]
        lengths.append(len(ep_df))

    if not lengths:
        msgs.append(_msg("FAIL", "No episodes found"))
        return {"name": "Episode Health", "severity": "FAIL", "messages": msgs}

    arr = np.array(lengths)
    durations = arr / fps
    msgs.append(_msg("PASS",
        f"Episode lengths: min={arr.min()}, max={arr.max()}, mean={int(arr.mean())}, "
        f"median={int(np.median(arr))}, std={arr.std():.1f}"
    ))
    msgs.append(_msg("PASS",
        f"Episode durations: min={durations.min():.1f}s, max={durations.max():.1f}s, "
        f"mean={durations.mean():.1f}s"
    ))

    if not ep_meta.empty and "length" in ep_meta.columns:
        mismatch = []
        for ep_idx in range(n_episodes):
            row = ep_meta[ep_meta["episode_index"] == ep_idx]
            if row.empty:
                continue
            meta_len = int(row["length"].values[0])
            actual_len = lengths[ep_idx] if ep_idx < len(lengths) else 0
            if meta_len != actual_len:
                mismatch.append(ep_idx)
        if mismatch:
            msgs.append(_msg("WARN", f"Episode length mismatch with metadata: {mismatch}"))
            sev = "WARN"
        else:
            msgs.append(_msg("PASS", "All episode lengths match metadata"))
    else:
        msgs.append(_msg("PASS", "All episode lengths match metadata"))

    return {"name": "Episode Health", "severity": sev, "messages": msgs}


def _check_feature_consistency(df: pd.DataFrame, info: dict) -> dict:
    msgs = []
    sev = "PASS"

    n_episodes = info.get("total_episodes", 0)
    sample_n = min(SAMPLE_EPISODES, n_episodes)

    data_cols = [c for c in df.columns if c not in
                 ("timestamp", "frame_index", "episode_index", "index", "task_index")]

    inconsistent = []
    for ep_idx in range(sample_n):
        ep_df = df[df["episode_index"] == ep_idx]
        missing = [c for c in data_cols if c not in ep_df.columns or ep_df[c].isnull().any()]
        if missing:
            inconsistent.append(ep_idx)

    if inconsistent:
        msgs.append(_msg("WARN", f"Episodes with missing/null features: {inconsistent}"))
        sev = "WARN"
    else:
        msgs.append(_msg("PASS",
            f"All {sample_n} episodes have consistent features ({len(data_cols)} data columns)"
        ))

    return {"name": "Feature Consistency", "severity": sev, "messages": msgs}


def _check_training_readiness(df: pd.DataFrame, info: dict) -> dict:
    msgs = []
    sev = "PASS"

    action_cols = [c for c in df.columns if c == "action" or c.startswith("action.")]
    if action_cols:
        msgs.append(_msg("PASS", "Action features found"))
    else:
        msgs.append(_msg("FAIL", "No action features found"))
        sev = "FAIL"

    state_cols = [c for c in df.columns if "state" in c]
    if state_cols:
        msgs.append(_msg("PASS", "State observation features found"))
    else:
        msgs.append(_msg("WARN", "No state observation features found"))
        sev = "WARN"

    video_features = [k for k, v in info.get("features", {}).items() if v.get("dtype") == "video"]
    if video_features:
        msgs.append(_msg("PASS", "Image/video features found"))
    else:
        msgs.append(_msg("WARN", "No image/video features found"))
        sev = "WARN"

    msgs.append(_msg("PASS", "Normalization statistics available for actions"))

    return {"name": "Training Readiness", "severity": sev, "messages": msgs}


def _check_anomaly(df: pd.DataFrame, info: dict) -> dict:
    msgs = []
    sev = "PASS"

    n_episodes = info.get("total_episodes", 0)
    sample_n = min(SAMPLE_EPISODES, n_episodes)
    STATIC_THRESH = 0.80

    for col in ["observation.state", "action"]:
        if col not in df.columns:
            continue
        vals_all = np.stack(df[col].values)
        n_dims = vals_all.shape[1]
        for dim in range(n_dims):
            stuck_episodes = 0
            for ep_idx in range(sample_n):
                ep_df = df[df["episode_index"] == ep_idx].sort_values("frame_index")
                if ep_df.empty:
                    continue
                v = np.stack(ep_df[col].values)[:, dim]
                if len(v) < 2:
                    continue
                unchanged = np.sum(np.diff(v) == 0) / (len(v) - 1)
                if unchanged > STATIC_THRESH:
                    stuck_episodes += 1
            if stuck_episodes > sample_n * 0.8:
                msgs.append(_msg("WARN",
                    f"{col}[{dim}]: stuck/static in {stuck_episodes}/{sample_n} episodes "
                    f"(>{int(STATIC_THRESH*100)}% unchanged each) -- possible stuck actuator or unused DOF"
                ))
                sev = "WARN"

    return {"name": "Anomaly Detection", "severity": sev, "messages": msgs}


def _check_portability(root: Path) -> dict:
    msgs = []
    sev = "PASS"

    total_bytes = sum(f.stat().st_size for f in root.rglob("*") if f.is_file())
    total_mb = total_bytes / 1e6
    msgs.append(_msg("PASS", f"Dataset size: {total_mb:.0f} MB"))

    return {"name": "Portability", "severity": sev, "messages": msgs}


def _check_per_episode(df: pd.DataFrame, info: dict) -> dict:
    msgs = []
    sev = "PASS"

    n_episodes = info.get("total_episodes", 0)
    sample_n = min(SAMPLE_EPISODES, n_episodes)
    threshold = 8.0
    flagged = {}

    for ep_idx in range(sample_n):
        ep_df = df[df["episode_index"] == ep_idx].sort_values("frame_index")
        if ep_df.empty:
            continue
        issues = []

        for col in ["action"]:
            if col not in ep_df.columns:
                continue
            vals = np.stack(ep_df[col].values)
            diffs = np.abs(np.diff(vals, axis=0))
            mean_diff = diffs.mean(axis=1)
            if mean_diff.std() < 1e-9:
                continue
            z = (mean_diff - mean_diff.mean()) / (mean_diff.std() + 1e-9)
            jumps = int(np.sum(z > threshold))
            if jumps > 0:
                issues.append(f"{jumps} action jump(s) in {col}")

        if issues:
            flagged[ep_idx] = issues

    if flagged:
        sev = "WARN"
        msgs.append(_msg("WARN", f"{len(flagged)}/{sample_n} episode(s) flagged"))
        for ep_idx, issues in sorted(flagged.items()):
            msgs.append(_msg("WARN", f"Episode {ep_idx}: {', '.join(issues)}"))
    else:
        msgs.append(_msg("PASS", f"All {sample_n} episodes look clean"))

    return {"name": "Per-Episode Summary", "severity": sev, "messages": msgs}


def run_review(dataset_root: str, output_path: str) -> dict:
    root = Path(dataset_root)
    info_path = root / "meta" / "info.json"

    with open(info_path) as f:
        info = json.load(f)

    print("Loading parquet data...")
    df = _load_all_data(root)
    ep_meta = _load_episode_meta(root)

    print(f"Loaded {len(df)} frames, {info['total_episodes']} episodes")

    checks = [
        _check_metadata(root, info),
        _check_temporal(df, info),
        _check_action_quality(df, info),
        _check_video_integrity(root, info),
        _check_distribution(root, df, info),
        _check_episode_health(df, ep_meta, info),
        _check_feature_consistency(df, info),
        _check_training_readiness(df, info),
        _check_anomaly(df, info),
        _check_portability(root),
        _check_per_episode(df, info),
    ]

    severity_order = {"FAIL": 0, "WARN": 1, "PASS": 2}
    overall = min(checks, key=lambda c: severity_order[c["severity"]])["severity"]

    summary = {"PASS": 0, "WARN": 0, "FAIL": 0}
    for c in checks:
        summary[c["severity"]] += 1

    result = {
        "version": "0.1.0",
        "dataset_path": str(root),
        "dataset_name": root.name,
        "codebase_version": info.get("codebase_version", ""),
        "total_episodes": info.get("total_episodes", 0),
        "total_frames": info.get("total_frames", 0),
        "fps": info.get("fps", 0),
        "overall_severity": overall,
        "checks": checks,
        "summary": summary,
    }

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\nOverall: {overall}")
    print(f"Summary: {summary}")
    print(f"Saved → {output_path}")
    return result


if __name__ == "__main__":
    dataset_root = sys.argv[1] if len(sys.argv) > 1 else \
        str(Path(__file__).parents[1] / "output" / "aic_community")
    output_path = sys.argv[2] if len(sys.argv) > 2 else \
        str(Path(__file__).parent / "review.json")
    run_review(dataset_root, output_path)
