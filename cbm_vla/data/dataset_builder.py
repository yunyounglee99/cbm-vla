"""
SO-101 Concept Dataset Builder
================================

HuggingFace LeRobot community dataset (v2/v3)를 로드하여
SO-100/SO-101 호환 데이터셋(action_dim=6)을 필터링하고,
auto_segmentation → llm_annotation → t5_refinement
전체 파이프라인을 실행하여 CBM-VLA 학습용 concept 데이터셋을 구축합니다.

=== SO-100 / SO-101 스펙 ===
    - 5 arm joints: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll
    - 1 gripper: gripper
    - action_dim = 6, state_dim = 6
    - FPS: 보통 30
    - robot_type: "so100", "so100_follower", "so101", "so101_follower"

=== Pipeline 단계 ===
    1. HuggingFace Hub에서 SO-100/SO-101 LeRobot 데이터셋 검색 & 로드
    2. AutoSegmenter로 에피소드별 세그먼트 분할
    3. GeminiConceptAnnotator로 각 세그먼트에 concept annotation 생성
    4. ConceptRefiner로 raw concept → 정제된 concept pool 구축
    5. 최종 concept 데이터셋 저장 (학습 루프에서 로드 가능한 형태)

=== 출력 데이터 구조 ===
    {
        "concept_pool": [...],           # 정제된 concept 풀 (20~50개)
        "dataset": [...],                # 프레임별 concept mapping
        "dataset_info": {                # 메타데이터
            "source_repos": [...],
            "total_episodes": int,
            "total_frames": int,
            "robot_type": str,
            "action_dim": int,
        }
    }

Usage:
    python -m cbm_vla.data.so101_dataset_builder \\
        --repo_ids lerobot/svla_so101_pickplace lerobot/svla_so100_stacking \\
        --output_dir ./data/so101_concept_dataset \\
        --max_episodes_per_repo 100 \\
        --gemini_api_key YOUR_API_KEY
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

# CBM-VLA data modules
from .auto_segmentation import AutoSegmenter
from .llm_annotation import GeminiConceptAnnotator
from .t5_refinement import ConceptRefiner


# ================================================================
# SO-100 / SO-101 호환 데이터셋 레지스트리
# ================================================================

# LeRobot 공식 + 커뮤니티 데이터셋 중 SO-100/SO-101 호환
# robot_type 기준으로 자동 필터링하지만, 알려진 repo는 미리 등록
KNOWN_SO_REPOS = [
    # === LeRobot 공식 SmolVLA 데이터셋 ===
    "lerobot/svla_so101_pickplace",
    "lerobot/svla_so100_stacking",
    "lerobot/svla_so100_sorting",
    # === 커뮤니티에서 자주 사용하는 데이터셋 (예시) ===
    # 사용자가 추가로 등록 가능
]

# SO-100/SO-101 호환 robot_type 목록
SO_COMPATIBLE_ROBOT_TYPES = {
    "so100",
    "so100_follower",
    "so101",
    "so101_follower",
    "so_arm100",
}

# SO-100/SO-101 관절 이름 (6-DOF: 5 joints + 1 gripper)
SO101_JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

SO101_ACTION_DIM = 6   # 5 arm joints + 1 gripper
SO101_STATE_DIM = 6    # 동일

# ================================================================
# Multi-Embodiment 설정
# ================================================================
MAX_ACTION_DIM = 14  # ALOHA(14-DoF) 등 가장 큰 로봇 기준
MAX_STATE_DIM = 14   # 상태 차원도 동일하게 맞춤


@dataclass
class DatasetInfo:
    """데이터셋 메타 정보"""
    repo_id: str
    robot_type: str
    action_dim: int
    state_dim: int
    total_episodes: int
    total_frames: int
    fps: int
    codebase_version: str
    camera_keys: List[str] = field(default_factory=list)
    task_description: str = ""


# ================================================================
# 1. 데이터셋 검색 & 로드
# ================================================================

def fetch_dataset_info(repo_id: str) -> Optional[DatasetInfo]:
    """
    HuggingFace Hub에서 LeRobot 데이터셋의 meta/info.json을 읽어
    SO-100/SO-101 호환 여부를 확인합니다.
    
    Args:
        repo_id: HuggingFace dataset repo ID (예: "lerobot/svla_so101_pickplace")
        
    Returns:
        DatasetInfo if compatible, None otherwise
    """
    try:
        from huggingface_hub import hf_hub_download
        
        # meta/info.json 다운로드
        info_path = hf_hub_download(
            repo_id=repo_id,
            filename="meta/info.json",
            repo_type="dataset",
        )
        
        with open(info_path) as f:
            info = json.load(f)
        
        robot_type = info.get("robot_type", "unknown")
        features = info.get("features", {})
        
        # action dim 확인
        action_info = features.get("action", {})
        action_dim = action_info.get("shape", [0])
        if isinstance(action_dim, list):
            action_dim = action_dim[0]
        
        # state dim 확인
        state_info = features.get("observation.state", {})
        state_dim = state_info.get("shape", [0])
        if isinstance(state_dim, list):
            state_dim = state_dim[0]
        
        # 카메라 키 수집
        camera_keys = [
            k for k in features.keys()
            if k.startswith("observation.images")
        ]
        
        return DatasetInfo(
            repo_id=repo_id,
            robot_type=robot_type,
            action_dim=action_dim,
            state_dim=state_dim,
            total_episodes=info.get("total_episodes", 0),
            total_frames=info.get("total_frames", 0),
            fps=info.get("fps", 30),
            codebase_version=info.get("codebase_version", "unknown"),
            camera_keys=camera_keys,
        )
        
    except Exception as e:
        print(f"  [WARNING] Failed to fetch info for {repo_id}: {e}")
        return None


def is_so_compatible(info: DatasetInfo) -> bool:
    """SO-100/SO-101 호환 여부 판단"""
    # robot_type 기반 판단
    rt = info.robot_type.lower().replace("-", "").replace("_", "")
    for compatible in SO_COMPATIBLE_ROBOT_TYPES:
        if compatible.replace("_", "") in rt:
            return True
    
    # action_dim 기반 보조 판단 (SO-100/101은 6-DOF)
    if info.action_dim == SO101_ACTION_DIM:
        return True
    
    return False


def discover_so_datasets(
    repo_ids: Optional[List[str]] = None,
    search_hub: bool = False,
    max_search: int = 50,
) -> List[DatasetInfo]:
    """
    SO-100/SO-101 호환 데이터셋을 검색합니다.
    
    Args:
        repo_ids: 직접 지정한 repo ID 리스트 (없으면 KNOWN_SO_REPOS 사용)
        search_hub: True면 HF Hub에서 추가 검색 (API 사용)
        max_search: Hub 검색 시 최대 결과 수
        
    Returns:
        호환 데이터셋 정보 리스트
    """
    candidates = list(repo_ids or KNOWN_SO_REPOS)
    
    # Hub에서 추가 검색
    if search_hub:
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            
            # "so100" 또는 "so101" 태그로 검색
            for query in ["so100 lerobot", "so101 lerobot"]:
                results = api.list_datasets(
                    search=query,
                    limit=max_search,
                    sort="downloads",
                    direction=-1,
                )
                for ds in results:
                    if ds.id not in candidates:
                        candidates.append(ds.id)
            
            print(f"  Hub search found {len(candidates)} candidate datasets")
            
        except Exception as e:
            print(f"  [WARNING] Hub search failed: {e}")
    
    # 각 후보에 대해 info 확인
    compatible = []
    for repo_id in candidates:
        print(f"  Checking {repo_id}...", end=" ")
        info = fetch_dataset_info(repo_id)
        
        if info is None:
            print("SKIP (failed to fetch)")
            continue
        
        if is_so_compatible(info):
            print(f"OK (robot={info.robot_type}, action_dim={info.action_dim}, "
                  f"episodes={info.total_episodes}, frames={info.total_frames})")
            compatible.append(info)
        else:
            print(f"SKIP (robot={info.robot_type}, action_dim={info.action_dim})")
    
    return compatible


# ================================================================
# 2. 데이터 로드 & 추출
# ================================================================

def load_episodes_from_repo(
    repo_id: str,
    dataset_info: DatasetInfo,
    max_episodes: Optional[int] = None,
    cache_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    LeRobot 데이터셋에서 에피소드별 action, state, 이미지 경로를 추출합니다.
    
    Args:
        repo_id: HuggingFace dataset repo ID
        dataset_info: 사전 확인된 데이터셋 정보
        max_episodes: 최대 로드 에피소드 수
        cache_dir: 로컬 캐시 디렉토리
        
    Returns:
        {
            "episodes": [
                {
                    "episode_id": int,
                    "actions": np.ndarray [T, action_dim],
                    "states": np.ndarray [T, state_dim],
                    "task_description": str,
                    "num_frames": int,
                    "fps": int,
                    "dataset_root": str,   # 이미지 접근용 로컬 경로
                }
            ],
            "dataset_info": DatasetInfo,
        }
    """
    episodes = []
    
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        
        print(f"\n  Loading LeRobotDataset: {repo_id}")
        kwargs = {"repo_id": repo_id}
        if cache_dir:
            kwargs["root"] = cache_dir
        
        dataset = LeRobotDataset(**kwargs)
        
        # 에피소드 수 결정
        num_episodes = dataset_info.total_episodes
        if max_episodes:
            num_episodes = min(num_episodes, max_episodes)
        
        print(f"  Processing {num_episodes}/{dataset_info.total_episodes} episodes...")
        
        # 에피소드별 데이터 추출
        for ep_idx in range(num_episodes):
            if ep_idx % 50 == 0:
                print(f"    Episode {ep_idx}/{num_episodes}...")
            
            try:
                # LeRobot 최신 버전(0.5.0+) 및 다양한 포맷 호환 인덱스 추출 로직
                if hasattr(dataset, 'meta') and hasattr(dataset.meta, 'episodes'):
                    episodes_meta = dataset.meta.episodes
                    
                    # 'hasattr' 대신 딕셔너리/데이터셋 Key 존재 여부('in')로 확인합니다.
                    if "dataset_from_index" in episodes_meta:
                        from_idx = episodes_meta["dataset_from_index"][ep_idx]
                        to_idx = episodes_meta["dataset_to_index"][ep_idx]
                    elif "from_index" in episodes_meta:
                        from_idx = episodes_meta["from_index"][ep_idx]
                        to_idx = episodes_meta["to_index"][ep_idx]
                    elif "from" in episodes_meta:
                        from_idx = episodes_meta["from"][ep_idx]
                        to_idx = episodes_meta["to"][ep_idx]
                    else:
                        print(f"    [WARNING] Cannot find index keys in meta.episodes, skipping ep {ep_idx}")
                        continue
                else:
                    print(f"    [WARNING] Cannot find episode boundaries, skipping ep {ep_idx}")
                    continue
                
                # 안전장치: Tensor나 Numpy 형식일 경우 순수 정수(int)로 변환
                if hasattr(from_idx, 'item'): from_idx = from_idx.item()
                if hasattr(to_idx, 'item'): to_idx = to_idx.item()
                
                # 프레임별 데이터 수집
                actions = []
                states = []
                task_desc = ""
                
                for frame_idx in range(from_idx, to_idx):
                    item = dataset[frame_idx]
                    
                    # Action
                    action = item["action"]
                    if hasattr(action, 'numpy'):
                        action = action.numpy()
                    actions.append(action)
                    
                    # State (observation.state)
                    if "observation.state" in item:
                        state = item["observation.state"]
                        if hasattr(state, 'numpy'):
                            state = state.numpy()
                        states.append(state)
                    
                    # Task description (첫 프레임에서만)
                    if frame_idx == from_idx:
                        if "task" in item:
                            task_desc = item["task"]
                            if hasattr(task_desc, 'item'):
                                task_desc = str(task_desc)
                        elif "language_instruction" in item:
                            task_desc = item["language_instruction"]
                            if hasattr(task_desc, 'item'):
                                task_desc = str(task_desc)
                
                if not actions:
                    continue
                
                raw_actions_array = np.stack(actions)
                raw_states_array = np.stack(states) if states else np.zeros_like(raw_actions_array)
                
                # --- [핵심] Multi-Embodiment Zero-Padding ---
                actual_action_dim = raw_actions_array.shape[-1]
                actual_state_dim = raw_states_array.shape[-1]
                
                # Action 패딩
                action_pad_size = MAX_ACTION_DIM - actual_action_dim
                if action_pad_size > 0:
                    # (0,0): 프레임 축은 그대로 유지, (0, action_pad_size): 차원 축 뒤쪽에 0 추가
                    actions_array = np.pad(raw_actions_array, ((0, 0), (0, action_pad_size)), mode='constant', constant_values=0.0)
                else:
                    actions_array = raw_actions_array
                
                # State 패딩 (상태 차원도 맞춰주어야 모델에러가 안 남)
                state_pad_size = MAX_STATE_DIM - actual_state_dim
                if state_pad_size > 0:
                    states_array = np.pad(raw_states_array, ((0, 0), (0, state_pad_size)), mode='constant', constant_values=0.0)
                else:
                    states_array = raw_states_array
                    
                # --- [핵심] Action Mask 생성 ---
                # 실제 로봇의 관절이 있는 곳은 1.0, 패딩으로 채운 가짜 관절은 0.0
                action_mask = np.zeros(MAX_ACTION_DIM, dtype=np.float32)
                action_mask[:actual_action_dim] = 1.0
                
                # 데이터셋 로컬 경로 (이미지 접근용)
                dataset_root = None
                if hasattr(dataset, 'root'):
                    dataset_root = str(dataset.root)
                elif hasattr(dataset, 'videos_dir'):
                    dataset_root = str(Path(dataset.videos_dir).parent)
                
                episodes.append({
                    "episode_id": ep_idx,
                    "actions": actions_array,       # 패딩된 [T, 14] 텐서
                    "states": states_array,         # 패딩된 [T, 14] 텐서
                    "action_mask": action_mask,     # [14] 마스크 (예: [1,1,1,1,1,1,0,0,0,0,0,0,0,0])
                    "actual_action_dim": actual_action_dim, # 원래 차원 (예: 6)
                    "robot_type": dataset_info.robot_type,  # 로봇 종류 (예: "so101")
                    "task_description": task_desc,
                    "num_frames": len(actions),
                    "fps": dataset_info.fps,
                    "dataset_root": dataset_root,
                    "repo_id": repo_id,
                })
                
            except Exception as e:
                print(f"    [WARNING] Failed to load episode {ep_idx}: {e}")
                continue
        
        print(f"  Loaded {len(episodes)} episodes from {repo_id}")
        
    except ImportError:
        print("  [WARNING] lerobot not installed. Generating dummy data.")
        episodes = _generate_dummy_episodes(
            num_episodes=max_episodes or 10,
            action_dim=dataset_info.action_dim,
            repo_id=repo_id,
        )
    except Exception as e:
        print(f"  [ERROR] Failed to load dataset {repo_id}: {e}")
        print("  Falling back to dummy data.")
        episodes = _generate_dummy_episodes(
            num_episodes=max_episodes or 10,
            action_dim=dataset_info.action_dim,
            repo_id=repo_id,
        )
    
    return {
        "episodes": episodes,
        "dataset_info": dataset_info,
    }


def _generate_dummy_episodes(
    num_episodes: int,
    action_dim: int = 6,
    repo_id: str = "dummy",
) -> List[dict]:
    """lerobot 미설치 시 테스트용 더미 에피소드 생성"""
    print(f"  Generating {num_episodes} dummy episodes (action_dim={action_dim})...")
    
    tasks = [
        "Pick up the red cube and place it in the box",
        "Stack the blue block on the green block",
        "Sort objects by color into bins",
        "Grasp the yellow cylinder and move it right",
        "Pick up the small sphere and drop it in the cup",
    ]
    
    episodes = []
    for ep_idx in range(num_episodes):
        num_frames = np.random.randint(200, 500)
        actions = np.random.randn(num_frames, action_dim) * 0.1
        
        # 그리퍼 상태 시뮬레이션
        gripper = np.zeros(num_frames)
        t1 = num_frames // 3
        t2 = 2 * num_frames // 3
        gripper[t1:t2] = 1.0
        actions[:, -1] = gripper
        
        episodes.append({
            "episode_id": ep_idx,
            "actions": actions,
            "states": actions.copy(),
            "task_description": tasks[ep_idx % len(tasks)],
            "num_frames": num_frames,
            "fps": 30,
            "dataset_root": None,
            "repo_id": repo_id,
        })
    
    return episodes


# ================================================================
# 3. 통합 파이프라인
# ================================================================

class SO101ConceptDatasetBuilder:
    """
    SO-101용 CBM-VLA Concept Dataset Builder
    
    전체 파이프라인을 순차 실행합니다:
    1. 데이터셋 검색 & 로드
    2. Auto Segmentation
    3. LLM Annotation (Gemini)
    4. T5 Concept Refinement
    5. 최종 저장
    
    Args:
        output_dir: 출력 디렉토리
        gemini_api_key: Gemini API 키 (없으면 fallback annotation 사용)
        max_concepts: 최대 concept pool 크기
        similarity_threshold: T5 클러스터링 유사도 임계값
        min_segment_frames: 최소 세그먼트 길이 (프레임)
        gemini_model: Gemini 모델명
    """
    
    def __init__(
        self,
        output_dir: str = "./data/so101_concept_dataset",
        gemini_api_key: Optional[str] = None,
        max_concepts: int = 50,
        similarity_threshold: float = 0.85,
        min_segment_frames: int = 15,
        gemini_model: str = "gemini-2.0-flash",
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Sub-modules 초기화
        self.segmenter = AutoSegmenter(
            min_segment_frames=min_segment_frames,
        )
        
        self.annotator = GeminiConceptAnnotator(
            api_key=gemini_api_key,
            model_name=gemini_model,
        )
        
        self.refiner = ConceptRefiner(
            similarity_threshold=similarity_threshold,
            max_concepts=max_concepts,
            min_frequency=3,
        )
        
        # 중간 결과 캐시 경로
        self.cache_dir = self.output_dir / "cache"
        self.cache_dir.mkdir(exist_ok=True)
    
    def build(
        self,
        repo_ids: Optional[List[str]] = None,
        max_episodes_per_repo: Optional[int] = None,
        search_hub: bool = False,
        skip_annotation: bool = False,
        cache_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        전체 파이프라인 실행
        
        Args:
            repo_ids: 사용할 데이터셋 repo ID 리스트
            max_episodes_per_repo: repo당 최대 에피소드 수
            search_hub: HF Hub에서 추가 SO 데이터셋 검색 여부
            skip_annotation: True면 LLM annotation 건너뜀 (fallback 사용)
            cache_dir: 데이터셋 캐시 디렉토리
            
        Returns:
            최종 concept 데이터셋 dict
        """
        print("=" * 70)
        print("CBM-VLA SO-101 Concept Dataset Builder")
        print("=" * 70)
        
        # ============================================================
        # Step 1: 데이터셋 검색 & 호환성 확인
        # ============================================================
        print("\n[Step 1] Discovering SO-100/SO-101 compatible datasets...")
        
        compatible_datasets = discover_so_datasets(
            repo_ids=repo_ids,
            search_hub=search_hub,
        )
        
        if not compatible_datasets:
            print("  No compatible datasets found!")
            print("  Creating dummy dataset for testing...")
            compatible_datasets = [DatasetInfo(
                repo_id="dummy/so101_test",
                robot_type="so101_follower",
                action_dim=6,
                state_dim=6,
                total_episodes=20,
                total_frames=6000,
                fps=30,
                codebase_version="v2.1",
            )]
        
        print(f"\n  Found {len(compatible_datasets)} compatible datasets:")
        for ds in compatible_datasets:
            print(f"    - {ds.repo_id}: {ds.total_episodes} episodes, "
                  f"{ds.total_frames} frames, {ds.robot_type}")
        
        # ============================================================
        # Step 2: 에피소드 로드
        # ============================================================
        print("\n[Step 2] Loading episodes...")
        
        all_episodes = []
        source_repos = []
        
        for ds_info in compatible_datasets:
            result = load_episodes_from_repo(
                repo_id=ds_info.repo_id,
                dataset_info=ds_info,
                max_episodes=max_episodes_per_repo,
                cache_dir=cache_dir,
            )
            
            episodes = result["episodes"]
            if episodes:
                # 에피소드 ID를 전역적으로 고유하게 재부여
                offset = len(all_episodes)
                for ep in episodes:
                    ep["global_episode_id"] = offset + ep["episode_id"]
                
                all_episodes.extend(episodes)
                source_repos.append(ds_info.repo_id)
        
        total_frames = sum(ep["num_frames"] for ep in all_episodes)
        print(f"\n  Total loaded: {len(all_episodes)} episodes, {total_frames} frames")
        
        # ============================================================
        # Step 3: Auto Segmentation
        # ============================================================
        print("\n[Step 3] Running Auto Segmentation...")
        
        # 캐시 확인
        seg_cache_path = self.cache_dir / "segmentation_results.json"
        
        if seg_cache_path.exists():
            print(f"  Loading cached segmentation from {seg_cache_path}")
            with open(seg_cache_path) as f:
                all_segmentations = json.load(f)
        else:
            all_segmentations = []
            
            for ep in all_episodes:
                seg_result = self.segmenter.segment_episode(
                    actions=ep["actions"],
                    episode_id=ep["global_episode_id"],
                    task_description=ep["task_description"],
                    dataset_name=ep["repo_id"],
                    fps=ep["fps"],
                )
                all_segmentations.append(asdict(seg_result))
            
            # 캐시 저장
            with open(seg_cache_path, 'w') as f:
                json.dump(all_segmentations, f, indent=2, ensure_ascii=False)
            print(f"  Cached segmentation to {seg_cache_path}")
        
        total_segments = sum(len(s["segments"]) for s in all_segmentations)
        avg_segments = total_segments / max(len(all_segmentations), 1)
        print(f"  Total segments: {total_segments} "
              f"(avg {avg_segments:.1f} per episode)")
        
        # ============================================================
        # Step 4: LLM Annotation (Gemini)
        # ============================================================
        print("\n[Step 4] Running LLM Concept Annotation...")
        
        ann_cache_path = self.cache_dir / "raw_annotations.json"
        
        if ann_cache_path.exists():
            print(f"  Loading cached annotations from {ann_cache_path}")
            with open(ann_cache_path) as f:
                all_annotations = json.load(f)
        else:
            if skip_annotation:
                print("  Skipping LLM annotation (using fallback)...")
            
            all_annotations = self.annotator.annotate_segments(
                segments=all_segmentations,
                dataset_root=all_episodes[0].get("dataset_root") if all_episodes else None,
                output_path=str(ann_cache_path),
            )
            
            # 캐시 저장
            with open(ann_cache_path, 'w') as f:
                json.dump(all_annotations, f, indent=2, ensure_ascii=False)
            print(f"  Cached annotations to {ann_cache_path}")
        
        print(f"  Total annotations: {len(all_annotations)}")
        
        # ============================================================
        # Step 5: T5 Concept Refinement
        # ============================================================
        print("\n[Step 5] Running T5 Concept Refinement...")
        
        concept_pool, final_dataset = self.refiner.refine_and_build(
            raw_concepts=all_annotations,
            segments=all_segmentations,
        )
        
        print(f"  Concept pool size: {len(concept_pool)}")
        print(f"  Final dataset entries: {len(final_dataset)}")
        
        if concept_pool:
            print("\n  Top 10 concepts:")
            for c in concept_pool[:10]:
                print(f"    [{c['concept_id']:2d}] {c['name']:25s} "
                      f"| freq={c['frequency']:4d} ({c['frequency_pct']:5.1f}%)")
        
        # ============================================================
        # Step 6: 에피소드 원본 데이터와 concept 매핑 병합
        # ============================================================
        print("\n[Step 6] Merging concept labels with episode data...")
        
        # episode_id → concept entries 인덱스
        ep_concept_map = defaultdict(list)
        for entry in final_dataset:
            ep_concept_map[entry["episode_id"]].append(entry)
        
        # 최종 학습용 데이터 구조 구성
        training_entries = []
        
        for ep in all_episodes:
            ep_id = ep["global_episode_id"]
            concepts = ep_concept_map.get(ep_id, [])
            
            if not concepts:
                continue
            
            # 에피소드 내 concept 순서 정렬
            concepts.sort(key=lambda x: x["segment_id"])
            
            # 활성 concept ID 리스트 (binary vector 생성용)
            active_concept_ids = list(set(c["concept_id"] for c in concepts))
            
            # concept 순서 (order_in_episode 기반)
            concept_order = [c["concept_id"] for c in concepts]
            
            training_entries.append({
                "episode_id": ep_id,
                "repo_id": ep["repo_id"],
                "task_description": ep["task_description"],
                "num_frames": ep["num_frames"],
                "fps": ep["fps"],
                "action_dim": ep["actions"].shape[-1],
                # Concept 정보
                "active_concept_ids": active_concept_ids,
                "concept_order": concept_order,
                "segments": concepts,
                # 각 세그먼트의 프레임 범위 (학습 시 해당 프레임에 concept label 부여)
                "frame_to_concept": self._build_frame_concept_map(
                    concepts, ep["num_frames"], len(concept_pool)
                ),
            })
        
        print(f"  Training entries: {len(training_entries)}")
        
        # ============================================================
        # Step 7: 최종 저장
        # ============================================================
        print("\n[Step 7] Saving final dataset...")
        
        output = {
            "concept_pool": concept_pool,
            "training_entries": training_entries,
            "dataset_info": {
                "source_repos": source_repos,
                "total_episodes": len(all_episodes),
                "total_frames": total_frames,
                "total_training_entries": len(training_entries),
                "robot_type": "so101",
                "action_dim": SO101_ACTION_DIM,
                "state_dim": SO101_STATE_DIM,
                "num_concepts": len(concept_pool),
                "joint_names": SO101_JOINT_NAMES,
                "pipeline_version": "1.0",
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
        }
        
        # concept_pool 저장
        pool_path = self.output_dir / "concept_pool.json"
        with open(pool_path, 'w') as f:
            json.dump(concept_pool, f, indent=2, ensure_ascii=False)
        print(f"  Saved concept pool → {pool_path}")
        
        # training_entries 저장 (frame_to_concept은 별도 npy로)
        entries_for_json = []
        for entry in training_entries:
            e = {k: v for k, v in entry.items() if k != "frame_to_concept"}
            entries_for_json.append(e)
        
        entries_path = self.output_dir / "training_entries.json"
        with open(entries_path, 'w') as f:
            json.dump(entries_for_json, f, indent=2, ensure_ascii=False)
        print(f"  Saved training entries → {entries_path}")
        
        # frame-level concept labels 저장 (numpy)
        labels_dir = self.output_dir / "frame_labels"
        labels_dir.mkdir(exist_ok=True)
        
        for entry in training_entries:
            ep_id = entry["episode_id"]
            label_path = labels_dir / f"episode_{ep_id:06d}_concepts.npy"
            np.save(label_path, entry["frame_to_concept"])
        print(f"  Saved frame labels → {labels_dir}/")
        
        # dataset_info 저장
        info_path = self.output_dir / "dataset_info.json"
        with open(info_path, 'w') as f:
            json.dump(output["dataset_info"], f, indent=2, ensure_ascii=False)
        print(f"  Saved dataset info → {info_path}")
        
        print("\n" + "=" * 70)
        print("✓ Dataset build complete!")
        print(f"  Output directory: {self.output_dir}")
        print(f"  Concept pool: {len(concept_pool)} concepts")
        print(f"  Training entries: {len(training_entries)} episodes")
        print(f"  Action dim: {SO101_ACTION_DIM} (SO-101: 5 joints + 1 gripper)")
        print("=" * 70)
        
        return output
    
    def _build_frame_concept_map(
        self,
        concepts: List[dict],
        num_frames: int,
        num_concepts: int,
    ) -> np.ndarray:
        """
        프레임별 활성 concept binary vector를 생성합니다.
        
        Args:
            concepts: 에피소드 내 concept 리스트 (segment별)
            num_frames: 에피소드 총 프레임 수
            num_concepts: concept pool 크기
            
        Returns:
            [num_frames, num_concepts] binary array
                각 프레임에서 활성인 concept = 1.0, 비활성 = 0.0
        """
        labels = np.zeros((num_frames, num_concepts), dtype=np.float32)
        
        for concept_entry in concepts:
            concept_id = concept_entry.get("concept_id", 0)
            start = concept_entry.get("start_frame", 0)
            end = min(concept_entry.get("end_frame", num_frames), num_frames)
            
            if 0 <= concept_id < num_concepts and start < end:
                labels[start:end, concept_id] = 1.0
        
        return labels


# ================================================================
# CLI
# ================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Build CBM-VLA concept dataset from SO-101 LeRobot data"
    )
    
    parser.add_argument(
        "--repo_ids", nargs="+", default=None,
        help="HuggingFace dataset repo IDs to use. "
             "If not specified, uses known SO-100/SO-101 repos."
    )
    parser.add_argument(
        "--output_dir", type=str, default="./data/so101_concept_dataset",
        help="Output directory for the concept dataset"
    )
    parser.add_argument(
        "--max_episodes_per_repo", type=int, default=None,
        help="Max episodes to load per repo"
    )
    parser.add_argument(
        "--gemini_api_key", type=str, default=None,
        help="Gemini API key (or set GEMINI_API_KEY env var)"
    )
    parser.add_argument(
        "--gemini_model", type=str, default="gemini-2.0-flash",
        help="Gemini model to use for annotation"
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
        "--search_hub", action="store_true",
        help="Search HuggingFace Hub for additional SO datasets"
    )
    parser.add_argument(
        "--skip_annotation", action="store_true",
        help="Skip LLM annotation (use rule-based fallback)"
    )
    parser.add_argument(
        "--cache_dir", type=str, default=None,
        help="Local cache directory for datasets"
    )
    
    args = parser.parse_args()
    
    builder = SO101ConceptDatasetBuilder(
        output_dir=args.output_dir,
        gemini_api_key=args.gemini_api_key or os.environ.get("GEMINI_API_KEY"),
        max_concepts=args.max_concepts,
        similarity_threshold=args.similarity_threshold,
        gemini_model=args.gemini_model,
    )
    
    result = builder.build(
        repo_ids=args.repo_ids,
        max_episodes_per_repo=args.max_episodes_per_repo,
        search_hub=args.search_hub,
        skip_annotation=args.skip_annotation,
        cache_dir=args.cache_dir,
    )
    
    return result


if __name__ == "__main__":
    main()