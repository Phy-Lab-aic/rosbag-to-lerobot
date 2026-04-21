"""
Bad 에피소드(success=False) 제거 후 새 LeRobot v3 데이터셋 생성 + HF 업로드.

처리 흐름:
  1. episode 메타에서 good/bad 분류
  2. frame parquet 필터링 + re-index
  3. video: ffmpeg으로 good episode 구간 추출 후 재조합
  4. meta(info.json, stats.json, tasks.parquet, episodes) 갱신
  5. HuggingFace Hub에 private으로 업로드

Usage:
  python filter_dataset.py --src /path/to/dataset --dst /path/to/output --repo-id org/repo
  python filter_dataset.py --src /path/to/dataset --dst /path/to/output  # 업로드 없이
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

CAMS = ["cam_left", "cam_center", "cam_right"]


# ── 1. 에피소드 분류 ──────────────────────────────────────────────────────────

def load_episode_meta(src: Path) -> pd.DataFrame:
    ep_dir = src / "meta" / "episodes"
    parts = sorted(ep_dir.rglob("*.parquet"))
    return pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)


def get_good_episodes(ep_df: pd.DataFrame) -> pd.DataFrame:
    good = ep_df[ep_df["grade"] != "Bad"].copy().reset_index(drop=True)
    logger.info("전체 %d → good %d, bad %d",
                len(ep_df), len(good), len(ep_df) - len(good))
    return good


# ── 2. frame parquet 필터링 + re-index ──────────────────────────────────────

def filter_frame_parquets(src: Path, dst: Path,
                          good_old_idx: List[int]) -> pd.DataFrame:
    """good 에피소드만 남기고 episode_index / index 재번호."""
    good_set = set(good_old_idx)
    old_to_new = {old: new for new, old in enumerate(good_old_idx)}

    data_files = sorted((src / "data").rglob("*.parquet"))
    all_chunks = []
    for fp in data_files:
        df = pd.read_parquet(fp)
        df = df[df["episode_index"].isin(good_set)].copy()
        all_chunks.append(df)

    full = pd.concat(all_chunks, ignore_index=True)
    full["episode_index"] = full["episode_index"].map(old_to_new)
    full = full.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)
    full["index"] = np.arange(len(full), dtype=np.int64)

    out_dir = dst / "data" / "chunk-000"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "file-000.parquet"
    full.to_parquet(out_path, index=False)
    logger.info("frame parquet 저장: %d rows → %s", len(full), out_path)
    return full


# ── 3. 비디오 필터링 ─────────────────────────────────────────────────────────

def _ffmpeg_extract_segment(src_mp4: Path, dst_mp4: Path,
                             t_start: float, t_end: float) -> bool:
    duration = t_end - t_start
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{t_start:.6f}",
        "-t",  f"{duration:.6f}",
        "-i",  str(src_mp4),
        "-c",  "copy",
        "-avoid_negative_ts", "make_zero",
        str(dst_mp4),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        logger.warning("ffmpeg 실패: %s\n%s", dst_mp4.name, r.stderr[-500:])
        return False
    return True


def _ffmpeg_concat(segment_paths: List[Path], out_mp4: Path) -> bool:
    if len(segment_paths) == 1:
        shutil.copy2(segment_paths[0], out_mp4)
        return True
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as flist:
        for p in segment_paths:
            flist.write(f"file '{p}'\n")
        list_path = flist.name
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", list_path,
        "-c", "copy",
        str(out_mp4),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    os.unlink(list_path)
    if r.returncode != 0:
        logger.warning("ffmpeg concat 실패: %s\n%s", out_mp4.name, r.stderr[-500:])
        return False
    return True


def filter_videos(src: Path, dst: Path,
                  good_ep_df: pd.DataFrame,
                  old_to_new: dict,
                  n_good: int) -> pd.DataFrame:
    updated = good_ep_df.copy()

    for cam in CAMS:
        cam_key = f"observation.images.{cam}"
        col_c   = f"videos/{cam_key}/chunk_index"
        col_f   = f"videos/{cam_key}/file_index"
        col_ts  = f"videos/{cam_key}/from_timestamp"
        col_te  = f"videos/{cam_key}/to_timestamp"

        if col_c not in good_ep_df.columns:
            logger.info("  [%s] 컬럼 없음, 스킵", cam)
            continue

        out_cam_dir = dst / "videos" / cam_key / "chunk-000"
        out_cam_dir.mkdir(parents=True, exist_ok=True)

        total = len(good_ep_df)
        for new_idx, (_, row) in enumerate(good_ep_df.iterrows()):
            old_ep    = int(row["episode_index"])
            src_chunk = int(row[col_c])
            src_file  = int(row[col_f])
            t_start   = float(row[col_ts])
            t_end     = float(row[col_te])

            src_mp4 = src / "videos" / cam_key / f"chunk-{src_chunk:03d}" / f"file-{src_file:03d}.mp4"
            out_mp4 = out_cam_dir / f"file-{new_idx:03d}.mp4"

            if new_idx % 50 == 0:
                logger.info("  [%s] %d/%d", cam, new_idx + 1, total)

            ok = _ffmpeg_extract_segment(src_mp4, out_mp4, t_start, t_end)
            if not ok:
                logger.warning("  [%s] ep%d 세그먼트 추출 실패, 스킵", cam, old_ep)
                continue

            duration = t_end - t_start
            mask = updated["episode_index"] == old_ep
            updated.loc[mask, col_c]  = 0
            updated.loc[mask, col_f]  = new_idx
            updated.loc[mask, col_ts] = 0.0
            updated.loc[mask, col_te] = round(duration, 4)

        logger.info("  [%s] 비디오 처리 완료", cam)

    updated["episode_index"] = updated["episode_index"].map(old_to_new)
    updated = updated.sort_values("episode_index").reset_index(drop=True)

    cum = 0
    for i, row in updated.iterrows():
        updated.at[i, "dataset_from_index"] = cum
        cum += int(row["length"])
        updated.at[i, "dataset_to_index"] = cum - 1

    return updated


# ── 4. meta 갱신 ─────────────────────────────────────────────────────────────

def write_episode_meta(dst: Path, updated_ep_df: pd.DataFrame):
    out_dir = dst / "meta" / "episodes" / "chunk-000"
    out_dir.mkdir(parents=True, exist_ok=True)
    updated_ep_df.to_parquet(out_dir / "file-000.parquet", index=False)
    logger.info("episode meta 저장: %d rows", len(updated_ep_df))


def write_info_json(src: Path, dst: Path,
                    n_good_ep: int, n_good_frames: int):
    with open(src / "meta" / "info.json") as f:
        info = json.load(f)
    info["total_episodes"] = n_good_ep
    info["total_frames"]   = n_good_frames
    (dst / "meta").mkdir(parents=True, exist_ok=True)
    with open(dst / "meta" / "info.json", "w") as f:
        json.dump(info, f, indent=4)
    logger.info("info.json 갱신: %d ep, %d frames", n_good_ep, n_good_frames)


def write_stats_json(dst: Path, frame_df: pd.DataFrame):
    stats = {}
    for col, key in [("observation.state", "observation.state"), ("action", "action")]:
        if col not in frame_df.columns:
            continue
        vals = np.stack(frame_df[col].values).astype(np.float64)
        stats[key] = {
            "min":  vals.min(axis=0).tolist(),
            "max":  vals.max(axis=0).tolist(),
            "mean": vals.mean(axis=0).tolist(),
            "std":  vals.std(axis=0).tolist(),
        }
    with open(dst / "meta" / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    logger.info("stats.json 재계산 완료")


def copy_tasks(src: Path, dst: Path):
    shutil.copy2(src / "meta" / "tasks.parquet",
                 dst / "meta" / "tasks.parquet")


# ── 5. HuggingFace 업로드 ────────────────────────────────────────────────────

def push_to_hub(dst: Path, repo_id: str, private: bool = True):
    from huggingface_hub import HfApi
    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)
    try:
        api.create_repo(repo_id=repo_id, repo_type="dataset",
                        exist_ok=True, private=private)
        logger.info("HF repo 준비 (private=%s): %s", private, repo_id)
    except Exception as e:
        logger.warning("repo 생성 오류: %s", e)

    logger.info("업로드 시작: %s → %s", dst, repo_id)
    api.upload_folder(
        folder_path=str(dst),
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="filtered: removed bad (success=False) episodes",
    )
    logger.info("업로드 완료: https://huggingface.co/datasets/%s", repo_id)


# ── main ─────────────────────────────────────────────────────────────────────

def run_filter(src: Path, dst: Path, repo_id: str = "", private: bool = True):
    if dst.exists():
        logger.info("기존 출력 디렉토리 삭제: %s", dst)
        shutil.rmtree(dst)
    dst.mkdir(parents=True)

    ep_df  = load_episode_meta(src)
    good   = get_good_episodes(ep_df)
    good_old_idx = good["episode_index"].tolist()
    old_to_new   = {old: new for new, old in enumerate(good_old_idx)}
    n_good = len(good)

    frame_df = filter_frame_parquets(src, dst, good_old_idx)
    n_good_frames = len(frame_df)

    logger.info("비디오 세그먼트 추출 시작 (%d episodes × %d cams)",
                n_good, len(CAMS))
    updated_ep = filter_videos(src, dst, good, old_to_new, n_good)

    write_episode_meta(dst, updated_ep)
    write_info_json(src, dst, n_good, n_good_frames)
    write_stats_json(dst, frame_df)
    copy_tasks(src, dst)

    logger.info("필터링 완료: %d → %d 에피소드, %d 프레임",
                len(ep_df), n_good, n_good_frames)

    if repo_id:
        push_to_hub(dst, repo_id, private=private)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", required=True, type=Path,
                        help="Source LeRobot v3 dataset directory")
    parser.add_argument("--dst", required=True, type=Path,
                        help="Output directory for filtered dataset")
    parser.add_argument("--repo-id", default="",
                        help="HuggingFace repo ID (e.g. org/repo). Omit to skip upload.")
    parser.add_argument("--public", action="store_true",
                        help="Upload as public (default: private)")
    args = parser.parse_args()

    run_filter(
        src=args.src,
        dst=args.dst,
        repo_id=args.repo_id,
        private=not args.public,
    )


if __name__ == "__main__":
    main()
