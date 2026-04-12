"""
CBM-VLA Community Dataset Builder
===================================

SmolVLA가 학습에 사용한 정확히 동일한 데이터셋을 로드하여
auto_segmentation → llm_annotation → t5_refinement 파이프라인을 실행합니다.

=== SmolVLA 학습 데이터셋 (정확한 출처) ===
    - Community Dataset v1: HuggingFaceVLA/community_dataset_v1
        · 128개 데이터셋, 11,100 에피소드, 5.1M 프레임, 119.3 GB
        · 55명의 컨트리뷰터
    - Community Dataset v2: HuggingFaceVLA/community_dataset_v2
        · 340개 데이터셋, 6,300 에피소드, 5.0M 프레임, 59 GB
        · 117명의 컨트리뷰터
    총합: 468개 데이터셋, ~17,400 에피소드, ~10M 프레임

=== 데이터셋 구조 ===
    community_dataset_v1/
    ├── contributor_name/
    │   ├── dataset_name/
    │   │   ├── data/          # Parquet 파일
    │   │   ├── videos/        # MP4 파일
    │   │   └── meta/
    │   │       └── info.json  # 로봇 타입, action_dim 등
    │   └── ...
    └── ...

=== Multi-Embodiment 설계 ===
    - SO-100/SO-101 (6-DoF), Koch (6-DoF), ALOHA (14-DoF) 등 모두 포함
    - 가장 큰 로봇(14-DoF) 기준으로 zero-padding
    - action_mask로 실제 관절과 패딩 구분

=== 출력 데이터 구조 ===
    {
        "concept_pool": [...],       # 정제된 concept 풀 (20~50개)
        "training_entries": [...],   # 에피소드별 concept 매핑
        "dataset_info": {            # 메타데이터
            "source": "HuggingFaceVLA/community_dataset_v1+v2",
            "total_sub_datasets": int,
            "total_episodes": int,
            "total_frames": int,
        }
    }

Usage:
    # 전체 데이터셋 (SmolVLA 학습과 동일)
    python -m cbm_vla.data.dataset_builder \\
        --output_dir ./data/smolvla_concept_dataset \\
        --gemini_api_key YOUR_KEY \\
        --local_dir /path/to/downloaded/datasets

    # 소규모 테스트 (에피소드 수 제한)
    python -m cbm_vla.data.dataset_builder \\
        --output_dir ./data/test_dataset \\
        --max_episodes_per_sub_dataset 10 \\
        --skip_annotation
"""

import json
import os
import sys
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict, field
from collections import defaultdict
import time

from .auto_segmentation import AutoSegmenter
from .llm_annotation import GeminiConceptAnnotator
from .t5_refinement import ConceptRefiner


# ================================================================
# SmolVLA 학습 데이터셋 정보 (공식 출처)
# ================================================================

# SmolVLA pretraining에 사용된 정확한 HuggingFace 레포 ID
SMOLVLA_COMMUNITY_REPOS = [
    "HuggingFaceVLA/community_dataset_v1",  # 128 datasets, 11.1K episodes
    "HuggingFaceVLA/community_dataset_v2",  # 340 datasets, 6.3K  episodes
]

# Multi-Embodiment: 가장 큰 로봇(ALOHA 14-DoF) 기준 패딩
MAX_ACTION_DIM = 14
MAX_STATE_DIM  = 14

# Done flag: 항상 action 배열의 절대 마지막 차원
# layout: [joint_0 ... joint_{n-1} | pad ... pad | done]
#          ← actual_action_dim →   ← padding →    ↑ idx=14
# 로봇 DoF가 몇이든 done은 index 14에 고정
ACTION_DIM_WITH_DONE = MAX_ACTION_DIM + 1  # = 15

# 알려진 로봇 타입 → action_dim 매핑 (info.json에 없을 때 fallback)
KNOWN_ROBOT_ACTION_DIMS = {
    "so100": 6, "so100_follower": 6,
    "so101": 6, "so101_follower": 6,
    "so_arm100": 6,
    "koch": 6, "koch_follower": 6,
    "lekiwi": 6,
    "aloha": 14, "aloha_stationary": 14,
    "stretch": 3,
    "panda": 7, "franka": 7,
    "xarm": 6, "xarm6": 6, "xarm7": 7,
    "ur5": 6, "ur5e": 6,
    "gr1": 14,
}


@dataclass
class SubDatasetInfo:
    """community_dataset 내 개별 sub-dataset 정보"""
    # 식별자
    contributor: str            # 컨트리뷰터 이름
    dataset_name: str           # 데이터셋 이름
    community_version: str      # "v1" 또는 "v2"
    local_root: str             # 로컬 경로 (data/, videos/, meta/ 포함) - 없으면 빈 문자열

    # Hub 스트리밍용 정보 (로컬 data/ 없을 때 사용)
    hub_repo_id: str = ""       # e.g. "HuggingFaceVLA/community_dataset_v1"
    hub_path_prefix: str = ""   # e.g. "00ri/so100_battery"

    # 메타 정보 (meta/info.json에서 추출)
    robot_type: str = "unknown"
    action_dim: int = 6
    state_dim: int = 6
    total_episodes: int = 0
    total_frames: int = 0
    fps: int = 30
    codebase_version: str = "unknown"
    camera_keys: List[str] = field(default_factory=list)


# ================================================================
# 1. community_dataset 구조 탐색 및 sub-dataset 목록 추출
# ================================================================

def discover_sub_datasets_from_local(
    local_dir: str,
    community_version: str,
    max_sub_datasets: Optional[int] = None,
) -> List[SubDatasetInfo]:
    """
    로컬에 다운로드된 community_dataset에서 모든 sub-dataset을 탐색합니다.

    구조:
        local_dir/
        ├── contributor1/
        │   ├── dataset_name_1/
        │   │   ├── meta/info.json  ← 이게 있으면 LeRobot 데이터셋
        │   │   ├── data/
        │   │   └── videos/
        │   └── dataset_name_2/
        └── contributor2/
            └── dataset_name_3/

    Args:
        local_dir: 다운로드된 community_dataset 디렉토리 경로
        community_version: "v1" 또는 "v2"
        max_sub_datasets: 최대 sub-dataset 수 제한 (None = 전체)

    Returns:
        SubDatasetInfo 리스트
    """
    root = Path(local_dir)
    if not root.exists():
        print(f"  [ERROR] Directory not found: {local_dir}")
        return []

    sub_datasets = []

    # contributor 폴더 순회
    for contributor_dir in sorted(root.iterdir()):
        if not contributor_dir.is_dir():
            continue
        if contributor_dir.name.startswith('.'):
            continue

        contributor = contributor_dir.name

        # dataset 폴더 순회
        for dataset_dir in sorted(contributor_dir.iterdir()):
            if not dataset_dir.is_dir():
                continue
            if dataset_dir.name.startswith('.'):
                continue

            # meta/info.json 존재 확인 → LeRobot 데이터셋 여부 판단
            info_path = dataset_dir / "meta" / "info.json"
            if not info_path.exists():
                # 일부 구조는 data/ 폴더만 있을 수 있음 - 스킵
                continue

            # info.json 파싱
            try:
                with open(info_path) as f:
                    info = json.load(f)
            except Exception as e:
                print(f"  [WARNING] Cannot read {info_path}: {e}")
                continue

            robot_type = info.get("robot_type", "unknown")

            # action_dim 추출
            features = info.get("features", {})
            action_info = features.get("action", {})
            action_shape = action_info.get("shape", [6])
            action_dim = action_shape[0] if isinstance(action_shape, list) else action_shape

            # state_dim 추출
            state_info = features.get("observation.state", {})
            state_shape = state_info.get("shape", [action_dim])
            state_dim = state_shape[0] if isinstance(state_shape, list) else state_shape

            # 카메라 키
            camera_keys = [k for k in features if k.startswith("observation.images")]

            sub_datasets.append(SubDatasetInfo(
                contributor=contributor,
                dataset_name=dataset_dir.name,
                community_version=community_version,
                local_root=str(dataset_dir),
                hub_repo_id=f"HuggingFaceVLA/community_dataset_{community_version}",
                hub_path_prefix=f"{contributor}/{dataset_dir.name}",
                robot_type=robot_type,
                action_dim=int(action_dim),
                state_dim=int(state_dim),
                total_episodes=info.get("total_episodes", 0),
                total_frames=info.get("total_frames", 0),
                fps=info.get("fps", 30),
                codebase_version=info.get("codebase_version", "unknown"),
                camera_keys=camera_keys,
            ))

            if max_sub_datasets and len(sub_datasets) >= max_sub_datasets:
                print(f"  Reached max_sub_datasets limit: {max_sub_datasets}")
                return sub_datasets

    return sub_datasets


def discover_sub_datasets_from_hub(
    community_version: str,
    cache_dir: Optional[str] = None,
    max_sub_datasets: Optional[int] = None,
) -> List[SubDatasetInfo]:
    """
    HuggingFace Hub에서 직접 community_dataset의 파일 목록을 가져와
    sub-dataset 구조를 탐색합니다. (로컬 다운로드 없이 스트리밍)

    Args:
        community_version: "v1" 또는 "v2"
        cache_dir: HF 캐시 디렉토리
        max_sub_datasets: 최대 sub-dataset 수

    Returns:
        SubDatasetInfo 리스트 (local_root는 캐시 경로)
    """
    repo_id = f"HuggingFaceVLA/community_dataset_{community_version}"
    print(f"  Fetching file list from Hub: {repo_id}")

    try:
        from huggingface_hub import HfApi, hf_hub_download
        api = HfApi()

        # 레포의 전체 파일 목록 가져오기
        all_files = list(api.list_repo_files(repo_id=repo_id, repo_type="dataset"))

        # meta/info.json 파일만 추출 → sub-dataset 목록 파악
        info_files = [f for f in all_files if f.endswith("meta/info.json")]
        print(f"  Found {len(info_files)} sub-datasets in {repo_id}")

        sub_datasets = []

        for info_file in info_files:
            # 경로 파싱: contributor/dataset_name/meta/info.json
            parts = Path(info_file).parts
            if len(parts) < 4:
                continue

            contributor = parts[0]
            dataset_name = parts[1]

            # info.json 다운로드
            try:
                local_info_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=info_file,
                    repo_type="dataset",
                    cache_dir=cache_dir,
                )
            except Exception as e:
                print(f"    [WARNING] Cannot download {info_file}: {e}")
                continue

            # 다운로드된 경로에서 sub-dataset root 추정
            local_meta_dir = Path(local_info_path).parent
            local_root = str(local_meta_dir.parent)

            try:
                with open(local_info_path) as f:
                    info = json.load(f)
            except Exception as e:
                print(f"    [WARNING] Cannot parse {local_info_path}: {e}")
                continue

            robot_type = info.get("robot_type", "unknown")
            features = info.get("features", {})

            action_info = features.get("action", {})
            action_shape = action_info.get("shape", [6])
            action_dim = action_shape[0] if isinstance(action_shape, list) else action_shape

            state_info = features.get("observation.state", {})
            state_shape = state_info.get("shape", [action_dim])
            state_dim = state_shape[0] if isinstance(state_shape, list) else state_shape

            camera_keys = [k for k in features if k.startswith("observation.images")]

            sub_datasets.append(SubDatasetInfo(
                contributor=contributor,
                dataset_name=dataset_name,
                community_version=community_version,
                local_root="",           # Hub 모드에서는 비워둠 (data/ 없음)
                hub_repo_id=repo_id,
                hub_path_prefix=f"{contributor}/{dataset_name}",
                robot_type=robot_type,
                action_dim=int(action_dim),
                state_dim=int(state_dim),
                total_episodes=info.get("total_episodes", 0),
                total_frames=info.get("total_frames", 0),
                fps=info.get("fps", 30),
                codebase_version=info.get("codebase_version", "unknown"),
                camera_keys=camera_keys,
            ))

            if max_sub_datasets and len(sub_datasets) >= max_sub_datasets:
                print(f"  Reached max_sub_datasets limit: {max_sub_datasets}")
                break

        return sub_datasets

    except Exception as e:
        print(f"  [ERROR] Hub discovery failed: {e}")
        return []


# ================================================================
# 2. 개별 sub-dataset 로드
# ================================================================

def _read_parquet_to_dict(parquet_path_or_file) -> Optional[dict]:
    """
    parquet 파일을 dict로 읽습니다.
    로컬 Path 또는 pyarrow filesystem file object 모두 지원.
    """
    try:
        import pyarrow.parquet as pq
        table = pq.read_table(parquet_path_or_file)
        return table.to_pydict()
    except Exception as e:
        return None


def _parse_episode_dict(
    df: dict,
    ep_num: int,
    global_episode_offset: int,
    sub_ds: SubDatasetInfo,
    dataset_root: Optional[str],
) -> Optional[dict]:
    """
    parquet dict → 에피소드 딕셔너리로 변환 (padding 포함)
    """
    if "action" not in df or len(df["action"]) == 0:
        return None

    actions_np = np.array(df["action"], dtype=np.float32)
    if actions_np.ndim == 1:
        actions_np = actions_np.reshape(-1, 1)
    actual_action_dim = actions_np.shape[-1]

    state_key = "observation.state"
    if state_key in df and len(df[state_key]) > 0:
        states_np = np.array(df[state_key], dtype=np.float32)
        if states_np.ndim == 1:
            states_np = states_np.reshape(-1, 1)
        actual_state_dim = states_np.shape[-1]
    else:
        states_np = np.zeros_like(actions_np)
        actual_state_dim = actual_action_dim

    # task description
    task_desc = ""
    for task_key in ["task", "language_instruction", "task_description"]:
        if task_key in df and len(df[task_key]) > 0:
            val = df[task_key][0]
            if isinstance(val, bytes):
                task_desc = val.decode("utf-8", errors="ignore")
            elif isinstance(val, str):
                task_desc = val
            else:
                task_desc = str(val)
            break

    # ── zero-padding (joint 차원만, done 제외) ──────────────────────────
    def pad_to(arr, target_dim):
        d = arr.shape[-1]
        if d < target_dim:
            return np.pad(arr, ((0, 0), (0, target_dim - d)))
        return arr[:, :target_dim]

    T = len(actions_np)
    actions_padded = pad_to(actions_np, MAX_ACTION_DIM)   # [T, 14]
    states_padded  = pad_to(states_np,  MAX_STATE_DIM)    # [T, 14]

    # ── done flag: 항상 action 배열의 절대 마지막 차원(index 14) ─────────
    # 에피소드 마지막 프레임만 1.0, 나머지 0.0
    # 최종 layout: [j0...j_{n-1} | pad... | done]   shape=[T, 15]
    # 로봇 DoF가 몇이든 done은 항상 index=MAX_ACTION_DIM 에 고정
    done_col = np.zeros((T, 1), dtype=np.float32)
    done_col[-1, 0] = 1.0
    actions_with_done = np.concatenate([actions_padded, done_col], axis=1)  # [T, 15]

    # ── action_mask [15]: 실제 joint=1, padding=0, done=1 ────────────────
    action_mask = np.zeros(ACTION_DIM_WITH_DONE, dtype=np.float32)
    action_mask[:min(actual_action_dim, MAX_ACTION_DIM)] = 1.0
    action_mask[-1] = 1.0   # done flag 는 항상 유효

    # ── actions_raw: segmentation 전용 (패딩·done 없이 실제 관절만) ──────
    raw_actions_for_seg = actions_np[:, :actual_action_dim]   # [T, actual_action_dim]

    return {
        "global_episode_id":  global_episode_offset + ep_num,
        "local_episode_id":   ep_num,
        "contributor":         sub_ds.contributor,
        "dataset_name":        sub_ds.dataset_name,
        "community_version":   sub_ds.community_version,
        "repo_id":             f"{sub_ds.contributor}/{sub_ds.dataset_name}",
        "robot_type":          sub_ds.robot_type,
        "task_description":    task_desc,
        "actions":             actions_with_done,    # [T, 15]  joints+pad+done
        "actions_raw":         raw_actions_for_seg,  # [T, D]   segmentation 전용
        "states":              states_padded,          # [T, 14]
        "action_mask":         action_mask,            # [15]  joint=1,pad=0,done=1
        "actual_action_dim":   actual_action_dim,
        "actual_state_dim":    actual_state_dim,
        "num_frames":          T,
        "fps":                 sub_ds.fps,
        "dataset_root":        dataset_root,
        "camera_keys":         sub_ds.camera_keys,
    }


def _collect_episode_parquet_files_local(data_dir: Path) -> Dict[int, Path]:
    """로컬 data/ 디렉토리에서 episode_*.parquet 파일 수집"""
    episode_files = {}
    for pf in sorted(data_dir.rglob("*.parquet")):
        name = pf.stem
        if name.startswith("episode_"):
            try:
                ep_num = int(name.replace("episode_", ""))
                episode_files[ep_num] = pf
            except ValueError:
                continue
    return episode_files


def _collect_episode_parquet_files_hub(
    hub_repo_id: str,
    hub_path_prefix: str,
) -> Dict[int, str]:
    """
    HfFileSystem으로 Hub의 data/ 경로에서 episode_*.parquet 목록 수집.
    반환값: {ep_num: hf_path}
    """
    try:
        from huggingface_hub import HfFileSystem
        fs = HfFileSystem()

        # HF path 형식: datasets/HuggingFaceVLA/community_dataset_v1/contributor/dataset/data/
        base = f"datasets/{hub_repo_id}/{hub_path_prefix}/data"

        # glob으로 parquet 파일 탐색 (chunk 하위 포함)
        patterns = [
            f"{base}/*.parquet",
            f"{base}/**/*.parquet",
        ]
        found = []
        for pat in patterns:
            try:
                found.extend(fs.glob(pat))
            except Exception:
                continue

        episode_files = {}
        for hf_path in sorted(found):
            name = Path(hf_path).stem
            if name.startswith("episode_"):
                try:
                    ep_num = int(name.replace("episode_", ""))
                    episode_files[ep_num] = hf_path
                except ValueError:
                    continue

        return episode_files

    except Exception as e:
        print(f"    [WARNING] HfFileSystem glob failed: {e}")
        return {}


def load_episodes_from_sub_dataset(
    sub_ds: SubDatasetInfo,
    max_episodes: Optional[int] = None,
    global_episode_offset: int = 0,
) -> List[dict]:
    """
    개별 sub-dataset(LeRobot 형식)에서 에피소드를 로드합니다.

    로컬 data/ 디렉토리가 있으면 로컬에서 읽고,
    없으면 HfFileSystem으로 Hub에서 직접 스트리밍 읽습니다.

    Args:
        sub_ds: SubDatasetInfo
        max_episodes: 최대 로드 에피소드 수
        global_episode_offset: 전역 에피소드 ID offset

    Returns:
        에피소드 딕셔너리 리스트
    """
    try:
        import pyarrow.parquet as pq
    except ImportError:
        print("    [WARNING] pyarrow not installed: pip install pyarrow")
        return _fallback_dummy_episodes(sub_ds, max_episodes or 5, global_episode_offset)

    # ── 로컬 vs Hub 분기 ──────────────────────────────────────────
    use_local = bool(sub_ds.local_root) and (Path(sub_ds.local_root) / "data").exists()
    use_hub   = bool(sub_ds.hub_repo_id) and bool(sub_ds.hub_path_prefix)

    if use_local:
        # ── 로컬 읽기 ────────────────────────────────────────────
        data_dir = Path(sub_ds.local_root) / "data"
        episode_files = _collect_episode_parquet_files_local(data_dir)

        if not episode_files:
            return []

        ep_nums = sorted(episode_files.keys())
        if max_episodes:
            ep_nums = ep_nums[:max_episodes]

        videos_dir = Path(sub_ds.local_root) / "videos"
        dataset_root = str(sub_ds.local_root) if videos_dir.exists() else None

        episodes = []
        for ep_num in ep_nums:
            df = _read_parquet_to_dict(episode_files[ep_num])
            if df is None:
                continue
            ep = _parse_episode_dict(df, ep_num, global_episode_offset, sub_ds, dataset_root)
            if ep:
                episodes.append(ep)
        return episodes

    elif use_hub:
        # ── Hub 스트리밍 읽기 (HfFileSystem) ────────────────────
        try:
            from huggingface_hub import HfFileSystem
            fs = HfFileSystem()
        except ImportError:
            print("    [WARNING] huggingface_hub not installed")
            return _fallback_dummy_episodes(sub_ds, max_episodes or 5, global_episode_offset)

        episode_files = _collect_episode_parquet_files_hub(
            hub_repo_id=sub_ds.hub_repo_id,
            hub_path_prefix=sub_ds.hub_path_prefix,
        )

        if not episode_files:
            return []

        ep_nums = sorted(episode_files.keys())
        if max_episodes:
            ep_nums = ep_nums[:max_episodes]

        episodes = []
        for ep_num in ep_nums:
            hf_path = episode_files[ep_num]
            try:
                with fs.open(hf_path, "rb") as f:
                    import pyarrow.parquet as pq
                    table = pq.read_table(f)
                    df = table.to_pydict()
            except Exception as e:
                print(f"    [WARNING] Cannot read {hf_path}: {e}")
                continue

            ep = _parse_episode_dict(df, ep_num, global_episode_offset, sub_ds, dataset_root=None)
            if ep:
                episodes.append(ep)

        return episodes

    else:
        return []


def _fallback_dummy_episodes(
    sub_ds: SubDatasetInfo,
    num_episodes: int,
    global_offset: int,
) -> List[dict]:
    """pyarrow 미설치 시 테스트용 더미 에피소드"""
    episodes = []
    action_dim = sub_ds.action_dim

    for i in range(num_episodes):
        num_frames = np.random.randint(150, 400)
        actions = np.random.randn(num_frames, action_dim).astype(np.float32) * 0.1

        # gripper 시뮬레이션 (마지막 차원)
        t1, t2 = num_frames // 3, 2 * num_frames // 3
        actions[t1:t2, -1] = 1.0

        # padding (joint 차원만)
        pad = MAX_ACTION_DIM - action_dim
        if pad > 0:
            actions_padded = np.pad(actions, ((0, 0), (0, pad)))
        else:
            actions_padded = actions[:, :MAX_ACTION_DIM]

        # done flag 추가 (마지막 프레임 = 1.0)
        done_col = np.zeros((num_frames, 1), dtype=np.float32)
        done_col[-1, 0] = 1.0
        actions_with_done = np.concatenate([actions_padded, done_col], axis=1)  # [T, 15]

        action_mask = np.zeros(ACTION_DIM_WITH_DONE, dtype=np.float32)
        action_mask[:min(action_dim, MAX_ACTION_DIM)] = 1.0
        action_mask[-1] = 1.0   # done flag

        episodes.append({
            "global_episode_id": global_offset + i,
            "local_episode_id": i,
            "contributor": sub_ds.contributor,
            "dataset_name": sub_ds.dataset_name,
            "community_version": sub_ds.community_version,
            "repo_id": f"{sub_ds.contributor}/{sub_ds.dataset_name}",
            "robot_type": sub_ds.robot_type,
            "task_description": f"Manipulation task by {sub_ds.contributor}",
            "actions":     actions_with_done,    # [T, 15]
            "actions_raw": actions,              # [T, action_dim] segmentation 전용
            "states":      actions_padded.copy(),
            "action_mask": action_mask,          # [15]
            "actual_action_dim": action_dim,
            "actual_state_dim":  action_dim,
            "num_frames":  num_frames,
            "fps":         sub_ds.fps,
            "dataset_root": None,
            "camera_keys": sub_ds.camera_keys,
        })

    return episodes


# ================================================================
# 3. 통합 파이프라인
# ================================================================

class SmolVLACommunityDatasetBuilder:
    """
    SmolVLA 학습에 사용된 정확히 동일한 데이터셋으로
    CBM-VLA Concept Dataset를 구축하는 통합 빌더.

    학습 데이터:
        - HuggingFaceVLA/community_dataset_v1 (128 datasets, 11.1K episodes)
        - HuggingFaceVLA/community_dataset_v2 (340 datasets, 6.3K  episodes)
        - 총 468 datasets, ~17,400 episodes, ~10M frames

    Args:
        output_dir: 출력 디렉토리
        gemini_api_key: Gemini API 키 (또는 OPENROUTER_API_KEY 환경변수)
        max_concepts: 최대 concept pool 크기
        similarity_threshold: T5 클러스터링 유사도 임계값
        min_segment_frames: 최소 세그먼트 길이 (프레임)
        gemini_model: 모델명 (기본: google/gemini-2.0-flash)
    """

    def __init__(
        self,
        output_dir: str = "./data/smolvla_concept_dataset",
        gemini_api_key: Optional[str] = None,
        max_concepts: int = 50,
        similarity_threshold: float = 0.85,
        min_segment_frames: int = 15,
        gemini_model: str = "google/gemini-2.0-flash",
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.segmenter = AutoSegmenter(min_segment_frames=min_segment_frames)

        self.annotator = GeminiConceptAnnotator(
            api_key=gemini_api_key or os.environ.get("OPENROUTER_API_KEY"),
            model_name=gemini_model,
        )

        self.refiner = ConceptRefiner(
            similarity_threshold=similarity_threshold,
            max_concepts=max_concepts,
            min_frequency=3,
        )

        self.cache_dir = self.output_dir / "cache"
        self.cache_dir.mkdir(exist_ok=True)

    # ----------------------------------------------------------
    def build(
        self,
        # 로컬 데이터셋 경로 (다운로드된 경우)
        v1_local_dir: Optional[str] = None,
        v2_local_dir: Optional[str] = None,
        # Hub에서 직접 탐색 (로컬 없을 때)
        use_hub: bool = True,
        hf_cache_dir: Optional[str] = None,
        # 제한 옵션
        max_sub_datasets: Optional[int] = None,
        max_episodes_per_sub_dataset: Optional[int] = None,
        # 기타
        skip_annotation: bool = False,
    ) -> Dict[str, Any]:
        """
        전체 파이프라인 실행

        Args:
            v1_local_dir: community_dataset_v1 로컬 경로
                          (예: "/data/community_dataset_v1")
            v2_local_dir: community_dataset_v2 로컬 경로
                          (예: "/data/community_dataset_v2")
            use_hub: True면 로컬 없을 때 HF Hub에서 직접 탐색
            hf_cache_dir: HF 캐시 디렉토리
            max_sub_datasets: 각 버전당 최대 sub-dataset 수
                               (None = 전체 / v1: 128, v2: 340)
            max_episodes_per_sub_dataset: sub-dataset당 최대 에피소드 수
                                          (None = 전체)
            skip_annotation: True면 LLM annotation 스킵 (fallback 사용)

        Returns:
            최종 concept 데이터셋 dict
        """
        print("=" * 70)
        print("CBM-VLA Community Dataset Builder")
        print("(SmolVLA Pretraining Dataset: v1 + v2)")
        print("=" * 70)

        # ============================================================
        # Step 1: Sub-dataset 목록 수집
        # ============================================================
        print("\n[Step 1] Discovering sub-datasets...")
        print(f"  Sources: HuggingFaceVLA/community_dataset_v1 + v2")

        all_sub_datasets: List[SubDatasetInfo] = []

        for version, local_dir in [("v1", v1_local_dir), ("v2", v2_local_dir)]:
            repo_id = f"HuggingFaceVLA/community_dataset_{version}"
            expected = 128 if version == "v1" else 340

            if local_dir and Path(local_dir).exists():
                # 로컬 탐색 (빠름)
                print(f"\n  [{version}] Scanning local: {local_dir}")
                sub_ds_list = discover_sub_datasets_from_local(
                    local_dir=local_dir,
                    community_version=version,
                    max_sub_datasets=max_sub_datasets,
                )
            elif use_hub:
                # Hub 탐색 (느리지만 다운로드 불필요)
                print(f"\n  [{version}] Scanning Hub: {repo_id}")
                sub_ds_list = discover_sub_datasets_from_hub(
                    community_version=version,
                    cache_dir=hf_cache_dir,
                    max_sub_datasets=max_sub_datasets,
                )
            else:
                print(f"\n  [{version}] Skipping (no local dir, use_hub=False)")
                continue

            found = len(sub_ds_list)
            print(f"  [{version}] Found {found}/{expected} sub-datasets")
            all_sub_datasets.extend(sub_ds_list)

        if not all_sub_datasets:
            print("\n  [WARNING] No sub-datasets found. Generating dummy data...")
            all_sub_datasets = self._generate_dummy_sub_datasets(10)

        # 통계 출력
        total_eps_meta = sum(ds.total_episodes for ds in all_sub_datasets)
        robot_types = set(ds.robot_type for ds in all_sub_datasets)
        print(f"\n  Total sub-datasets: {len(all_sub_datasets)}")
        print(f"  Expected episodes (from meta): {total_eps_meta:,}")
        print(f"  Robot types: {', '.join(sorted(robot_types))}")

        # sub-dataset 목록 저장 (재현성)
        sub_ds_list_path = self.cache_dir / "sub_dataset_list.json"
        with open(sub_ds_list_path, 'w') as f:
            json.dump([asdict(ds) for ds in all_sub_datasets], f, indent=2)
        print(f"  Saved sub-dataset list → {sub_ds_list_path}")

        # ============================================================
        # Step 2: 에피소드 로드
        # ============================================================
        print("\n[Step 2] Loading episodes from all sub-datasets...")

        all_episodes = []
        load_stats = {"success": 0, "failed": 0, "total_frames": 0}

        for i, sub_ds in enumerate(all_sub_datasets):
            if i % 20 == 0:
                print(f"  Processing sub-dataset {i+1}/{len(all_sub_datasets)}: "
                      f"{sub_ds.contributor}/{sub_ds.dataset_name} "
                      f"(robot={sub_ds.robot_type}, action_dim={sub_ds.action_dim})")

            try:
                eps = load_episodes_from_sub_dataset(
                    sub_ds=sub_ds,
                    max_episodes=max_episodes_per_sub_dataset,
                    global_episode_offset=len(all_episodes),
                )

                if eps:
                    all_episodes.extend(eps)
                    load_stats["success"] += 1
                    load_stats["total_frames"] += sum(e["num_frames"] for e in eps)
                else:
                    load_stats["failed"] += 1

            except Exception as e:
                print(f"    [ERROR] {sub_ds.contributor}/{sub_ds.dataset_name}: {e}")
                load_stats["failed"] += 1
                continue

        print(f"\n  Loaded: {len(all_episodes):,} episodes, "
              f"{load_stats['total_frames']:,} frames")
        print(f"  Sub-datasets: {load_stats['success']} OK, "
              f"{load_stats['failed']} failed")

        if not all_episodes:
            raise RuntimeError("No episodes loaded. Check dataset paths.")

        # ============================================================
        # Step 3: Auto Segmentation
        # ============================================================
        print("\n[Step 3] Running Auto Segmentation...")

        seg_cache_path = self.cache_dir / "segmentation_results.json"

        if seg_cache_path.exists():
            print(f"  Loading cached segmentation from {seg_cache_path}")
            with open(seg_cache_path) as f:
                all_segmentations = json.load(f)
        else:
            all_segmentations = []
            for idx, ep in enumerate(all_episodes):
                if idx % 1000 == 0:
                    print(f"  Segmenting episode {idx}/{len(all_episodes)}...")

                # actions_raw: 패딩·done 제거한 실제 관절만 사용
                # → gripper 감지, 속도/방향 분석이 패딩 0값에 오염되지 않음
                seg_result = self.segmenter.segment_episode(
                    actions=ep["actions_raw"],
                    episode_id=ep["global_episode_id"],
                    task_description=ep["task_description"],
                    dataset_name=ep["repo_id"],
                    fps=ep["fps"],
                )
                # robot_type 정보 추가 (annotation에서 활용)
                seg_dict = asdict(seg_result)
                seg_dict["robot_type"] = ep["robot_type"]
                seg_dict["contributor"] = ep["contributor"]
                all_segmentations.append(seg_dict)

            with open(seg_cache_path, 'w') as f:
                json.dump(all_segmentations, f, indent=2, ensure_ascii=False)
            print(f"  Cached segmentation → {seg_cache_path}")

        total_segments = sum(len(s["segments"]) for s in all_segmentations)
        avg_per_ep = total_segments / max(len(all_segmentations), 1)
        print(f"  Total segments: {total_segments:,} "
              f"(avg {avg_per_ep:.1f}/episode)")

        # ============================================================
        # Step 4: LLM Annotation (Gemini via OpenRouter)
        # ============================================================
        print("\n[Step 4] Running LLM Concept Annotation...")

        ann_cache_path = self.cache_dir / "raw_annotations.json"

        if ann_cache_path.exists():
            print(f"  Loading cached annotations from {ann_cache_path}")
            with open(ann_cache_path) as f:
                all_annotations = json.load(f)
        else:
            # dataset_root 매핑 (episode_id → dataset_root)
            ep_root_map = {ep["global_episode_id"]: ep["dataset_root"]
                          for ep in all_episodes}
            # 첫 번째 유효한 dataset_root 사용 (annotator에 전달)
            first_valid_root = next(
                (ep["dataset_root"] for ep in all_episodes if ep["dataset_root"]),
                None
            )

            all_annotations = self.annotator.annotate_segments(
                segments=all_segmentations,
                dataset_root=first_valid_root,
                output_path=str(ann_cache_path),
            )

            with open(ann_cache_path, 'w') as f:
                json.dump(all_annotations, f, indent=2, ensure_ascii=False)
            print(f"  Cached annotations → {ann_cache_path}")

        print(f"  Total annotations: {len(all_annotations):,}")

        # ============================================================
        # Step 5: T5 Concept Refinement
        # ============================================================
        print("\n[Step 5] Running T5 Concept Refinement...")

        concept_pool, final_dataset = self.refiner.refine_and_build(
            raw_concepts=all_annotations,
            segments=all_segmentations,
        )

        print(f"  Concept pool: {len(concept_pool)} concepts")
        print(f"  Final dataset: {len(final_dataset):,} entries")

        if concept_pool:
            print("\n  Top 15 concepts:")
            for c in concept_pool[:15]:
                print(f"    [{c['concept_id']:2d}] {c['name']:25s} "
                      f"| freq={c['frequency']:5d} ({c['frequency_pct']:5.1f}%)"
                      f" | cluster_size={c['cluster_size']}")

        # ============================================================
        # Step 6: Concept ↔ Episode 매핑
        # ============================================================
        print("\n[Step 6] Merging concept labels with episode data...")

        ep_concept_map = defaultdict(list)
        for entry in final_dataset:
            ep_concept_map[entry["episode_id"]].append(entry)

        training_entries = []
        for ep in all_episodes:
            ep_id = ep["global_episode_id"]
            concepts = ep_concept_map.get(ep_id, [])
            if not concepts:
                continue

            concepts.sort(key=lambda x: x["segment_id"])
            active_concept_ids = list(set(c["concept_id"] for c in concepts))
            concept_order = [c["concept_id"] for c in concepts]

            training_entries.append({
                "episode_id": ep_id,
                "repo_id": ep["repo_id"],
                "contributor": ep["contributor"],
                "community_version": ep["community_version"],
                "robot_type": ep["robot_type"],
                "actual_action_dim": ep["actual_action_dim"],
                "action_mask": ep["action_mask"].tolist(),
                "task_description": ep["task_description"],
                "num_frames": ep["num_frames"],
                "fps": ep["fps"],
                "active_concept_ids": active_concept_ids,
                "concept_order": concept_order,
                "segments": concepts,
                "frame_to_concept": self._build_frame_concept_map(
                    concepts, ep["num_frames"], len(concept_pool)
                ),
            })

        print(f"  Training entries: {len(training_entries):,}")

        # ============================================================
        # Step 7: 최종 저장
        # ============================================================
        print("\n[Step 7] Saving final dataset...")

        # Robot type 통계
        robot_stats = defaultdict(int)
        action_dim_stats = defaultdict(int)
        for ep in all_episodes:
            robot_stats[ep["robot_type"]] += 1
            action_dim_stats[ep["actual_action_dim"]] += 1

        dataset_info = {
            "source_repos": SMOLVLA_COMMUNITY_REPOS,
            "total_sub_datasets": len(all_sub_datasets),
            "total_episodes": len(all_episodes),
            "total_frames": load_stats["total_frames"],
            "total_training_entries": len(training_entries),
            "total_segments": total_segments,
            "num_concepts": len(concept_pool),
            "max_action_dim": ACTION_DIM_WITH_DONE,   # 15 (14 joints + done)
            "max_state_dim": MAX_STATE_DIM,
            "robot_type_distribution": dict(robot_stats),
            "action_dim_distribution": {str(k): v for k, v in action_dim_stats.items()},
            "pipeline_version": "2.0",
            "smolvla_compatible": True,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        # concept_pool.json
        pool_path = self.output_dir / "concept_pool.json"
        with open(pool_path, 'w') as f:
            json.dump(concept_pool, f, indent=2, ensure_ascii=False)
        print(f"  Saved concept pool → {pool_path}")

        # training_entries.json (frame_to_concept 제외)
        entries_for_json = [
            {k: v for k, v in e.items() if k != "frame_to_concept"}
            for e in training_entries
        ]
        entries_path = self.output_dir / "training_entries.json"
        with open(entries_path, 'w') as f:
            json.dump(entries_for_json, f, indent=2, ensure_ascii=False)
        print(f"  Saved training entries → {entries_path}")

        # frame-level labels (numpy)
        labels_dir = self.output_dir / "frame_labels"
        labels_dir.mkdir(exist_ok=True)
        for entry in training_entries:
            ep_id = entry["episode_id"]
            label_path = labels_dir / f"episode_{ep_id:06d}_concepts.npy"
            np.save(label_path, entry["frame_to_concept"])
        print(f"  Saved frame labels → {labels_dir}/")

        # dataset_info.json
        info_path = self.output_dir / "dataset_info.json"
        with open(info_path, 'w') as f:
            json.dump(dataset_info, f, indent=2, ensure_ascii=False)
        print(f"  Saved dataset info → {info_path}")

        print("\n" + "=" * 70)
        print("✓ SmolVLA Community Dataset Build Complete!")
        print(f"  Sources: community_dataset_v1 + v2")
        print(f"  Sub-datasets: {len(all_sub_datasets)}")
        print(f"  Episodes: {len(all_episodes):,}")
        print(f"  Segments: {total_segments:,}")
        print(f"  Concepts: {len(concept_pool)}")
        print(f"  Training entries: {len(training_entries):,}")
        print(f"  Output: {self.output_dir}")
        print("=" * 70)

        return {
            "concept_pool": concept_pool,
            "training_entries": training_entries,
            "dataset_info": dataset_info,
        }

    # ----------------------------------------------------------
    def _build_frame_concept_map(
        self,
        concepts: List[dict],
        num_frames: int,
        num_concepts: int,
    ) -> np.ndarray:
        """프레임별 활성 concept binary vector [num_frames, num_concepts]"""
        labels = np.zeros((num_frames, num_concepts), dtype=np.float32)
        for c in concepts:
            cid = c.get("concept_id", 0)
            start = c.get("start_frame", 0)
            end = min(c.get("end_frame", num_frames), num_frames)
            if 0 <= cid < num_concepts and start < end:
                labels[start:end, cid] = 1.0
        return labels

    # ----------------------------------------------------------
    def _generate_dummy_sub_datasets(self, n: int) -> List[SubDatasetInfo]:
        """테스트용 더미 sub-dataset 목록"""
        dummy = []
        for i in range(n):
            dummy.append(SubDatasetInfo(
                contributor=f"contributor_{i:03d}",
                dataset_name=f"pick_place_{i:03d}",
                community_version="v1",
                local_root=f"/tmp/dummy_{i}",
                robot_type="so101" if i % 2 == 0 else "koch",
                action_dim=6,
                state_dim=6,
                total_episodes=20,
                total_frames=6000,
                fps=30,
                codebase_version="v2.1",
            ))
        return dummy


# ================================================================
# 비용 추정 유틸리티
# ================================================================

def estimate_annotation_cost(
    total_episodes: int = 17400,
    avg_segments_per_episode: float = 4.5,
    model: str = "gemini-2.0-flash",
) -> None:
    """SmolVLA 전체 데이터셋 annotation 비용 추정 출력"""
    from .llm_annotation import GeminiConceptAnnotator
    annotator = GeminiConceptAnnotator()

    total_segments = int(total_episodes * avg_segments_per_episode)
    cost = annotator.estimate_cost(total_segments, model)

    print("\n=== SmolVLA Full Dataset Annotation Cost Estimate ===")
    print(f"  Dataset: community_dataset_v1 + v2")
    print(f"  Episodes: {total_episodes:,}")
    print(f"  Avg segments/episode: {avg_segments_per_episode}")
    print(f"  Total segments: {total_segments:,}")
    print(f"  Model: {model}")
    print(f"  Input tokens: {cost['total_input_tokens']:,} → ${cost['input_cost_usd']:.2f}")
    print(f"  Output tokens: {cost['total_output_tokens']:,} → ${cost['output_cost_usd']:.2f}")
    print(f"  Total cost: ${cost['total_cost_usd']:.2f} USD (≈ ₩{cost['total_cost_usd']*1400:.0f})")
    print(f"  Batch 50% discount: ${cost['batch_discount_50pct']:.2f} USD")
    print("=" * 52)


# ================================================================
# CLI
# ================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Build CBM-VLA concept dataset from SmolVLA community datasets"
    )
    parser.add_argument(
        "--v1_local_dir", type=str, default=None,
        help="Local path to HuggingFaceVLA/community_dataset_v1 "
             "(e.g. /data/community_dataset_v1). "
             "Download: hf download HuggingFaceVLA/community_dataset_v1 "
             "--repo-type=dataset --local-dir /data/community_dataset_v1"
    )
    parser.add_argument(
        "--v2_local_dir", type=str, default=None,
        help="Local path to HuggingFaceVLA/community_dataset_v2"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./data/smolvla_concept_dataset",
        help="Output directory"
    )
    parser.add_argument(
        "--gemini_api_key", type=str, default=None,
        help="OpenRouter API key (or set OPENROUTER_API_KEY env var)"
    )
    parser.add_argument(
        "--gemini_model", type=str, default="google/gemini-2.0-flash",
        help="Model to use via OpenRouter (e.g. google/gemini-2.0-flash)"
    )
    parser.add_argument(
        "--max_sub_datasets", type=int, default=None,
        help="Max sub-datasets per community version (None = all)"
    )
    parser.add_argument(
        "--max_episodes_per_sub_dataset", type=int, default=None,
        help="Max episodes per sub-dataset (None = all)"
    )
    parser.add_argument(
        "--max_concepts", type=int, default=50,
        help="Maximum concept pool size"
    )
    parser.add_argument(
        "--similarity_threshold", type=float, default=0.85,
        help="T5 clustering similarity threshold"
    )
    parser.add_argument(
        "--use_hub", action="store_true", default=True,
        help="Use HF Hub when local dir not available"
    )
    parser.add_argument(
        "--hf_cache_dir", type=str, default=None,
        help="HuggingFace cache directory"
    )
    parser.add_argument(
        "--skip_annotation", action="store_true",
        help="Skip LLM annotation (use rule-based fallback)"
    )
    parser.add_argument(
        "--estimate_cost_only", action="store_true",
        help="Only print cost estimate and exit"
    )

    args = parser.parse_args()

    if args.estimate_cost_only:
        estimate_annotation_cost()
        return

    builder = SmolVLACommunityDatasetBuilder(
        output_dir=args.output_dir,
        gemini_api_key=args.gemini_api_key or os.environ.get("OPENROUTER_API_KEY"),
        max_concepts=args.max_concepts,
        similarity_threshold=args.similarity_threshold,
        gemini_model=args.gemini_model,
    )

    builder.build(
        v1_local_dir=args.v1_local_dir,
        v2_local_dir=args.v2_local_dir,
        use_hub=args.use_hub,
        hf_cache_dir=args.hf_cache_dir,
        max_sub_datasets=args.max_sub_datasets,
        max_episodes_per_sub_dataset=args.max_episodes_per_sub_dataset,
        skip_annotation=args.skip_annotation,
    )


if __name__ == "__main__":
    main()