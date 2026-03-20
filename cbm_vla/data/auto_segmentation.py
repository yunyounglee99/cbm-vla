"""
Stage 1: Rule-based Auto Segmentation
======================================

로봇 action 데이터를 분석하여 에피소드를 의미 있는 세그먼트로 자동 분할합니다.
사람의 annotation 없이 순수 코드로 동작합니다.

분할 기준 (3가지 신호의 OR 조합):
  1. Gripper state change: 그리퍼가 열림↔닫힘 전환 시점
  2. Velocity change: 전체 관절 속도의 급격한 변화 (정지→이동, 이동→정지)
  3. Direction change: 각 관절의 이동 방향이 반전되는 시점

각 세그먼트에서 대표 프레임(중간 지점)을 추출하여 LLM annotation에 사용합니다.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import json


@dataclass
class Segment:
    """단일 세그먼트 정보"""
    segment_id: int
    start_frame: int
    end_frame: int
    representative_frame: int
    trigger: str             # 분할을 유발한 신호 ("gripper" | "velocity" | "direction")
    duration_frames: int
    
    # 세그먼트 내 로봇 상태 요약 (LLM prompt에 활용)
    gripper_state: str       # "open" | "closed" | "transitioning"
    mean_velocity: float     # 평균 관절 속도
    dominant_direction: str  # "forward" | "backward" | "up" | "down" | "lateral" | "mixed"


@dataclass
class EpisodeSegmentation:
    """에피소드 단위 분할 결과"""
    episode_id: int
    task_description: str
    total_frames: int
    segments: List[dict]
    
    # 메타데이터
    dataset_name: str
    fps: int


class AutoSegmenter:
    """
    Rule-based Auto Segmenter
    
    LeRobot 데이터셋의 action 데이터를 분석하여
    에피소드를 3~6개의 의미 있는 세그먼트로 분할합니다.
    
    Args:
        gripper_threshold: 그리퍼 상태 변화 감지 임계값 (정규화된 값)
        velocity_threshold: 속도 변화 감지 임계값 (관절 각속도 기준)
        direction_change_threshold: 방향 변화 감지 코사인 유사도 임계값
        min_segment_frames: 최소 세그먼트 길이 (너무 짧은 세그먼트 병합)
    """
    
    def __init__(
        self,
        gripper_threshold: float = 0.5,
        velocity_threshold: float = 0.15,
        direction_change_threshold: float = 0.8,
        min_segment_frames: int = 15,
    ):
        self.gripper_threshold = gripper_threshold
        self.velocity_threshold = velocity_threshold
        self.direction_change_threshold = direction_change_threshold
        self.min_segment_frames = min_segment_frames
    
    def detect_gripper_changes(self, actions: np.ndarray) -> List[int]:
        """
        그리퍼 상태 변화 시점 감지
        
        SO-100 action format: [shoulder_pan, shoulder_lift, elbow_flex,
                               wrist_flex, wrist_roll, gripper]
        → 마지막 차원이 gripper
        
        Args:
            actions: [num_frames, action_dim] — action_dim의 마지막이 gripper
            
        Returns:
            change_points: 그리퍼 상태가 바뀌는 프레임 인덱스 리스트
        """
        gripper = actions[:, -1]  # 마지막 차원 = gripper
        
        # 그리퍼 값의 변화량 계산
        gripper_diff = np.abs(np.diff(gripper))
        
        # Threshold 초과 시점 감지
        change_points = []
        for i in range(len(gripper_diff)):
            if gripper_diff[i] > self.gripper_threshold:
                change_points.append(i + 1)  # diff 기준이므로 +1
        
        return change_points
    
    def detect_velocity_changes(self, actions: np.ndarray) -> List[int]:
        """
        관절 속도 급변 시점 감지
        
        전체 관절의 L2 속도를 계산하고, 속도 프로파일의
        급격한 변화 지점(정지→이동, 이동→정지)을 감지합니다.
        
        Args:
            actions: [num_frames, action_dim]
            
        Returns:
            change_points: 속도가 급변하는 프레임 인덱스 리스트
        """
        # 그리퍼 제외한 관절 속도 계산
        joint_actions = actions[:, :-1]  # gripper 제외
        velocities = np.diff(joint_actions, axis=0)  # [N-1, joint_dim]
        speed = np.linalg.norm(velocities, axis=1)   # [N-1]
        
        # 이동평균으로 스무딩 (노이즈 제거)
        window = 5
        if len(speed) > window:
            kernel = np.ones(window) / window
            speed_smooth = np.convolve(speed, kernel, mode='same')
        else:
            speed_smooth = speed
        
        # 속도 변화량
        speed_diff = np.abs(np.diff(speed_smooth))
        
        # 적응적 threshold: 전체 속도 범위 기준
        adaptive_threshold = self.velocity_threshold * (speed_smooth.max() - speed_smooth.min() + 1e-6)
        
        change_points = []
        for i in range(len(speed_diff)):
            if speed_diff[i] > adaptive_threshold:
                change_points.append(i + 2)  # diff 두 번이므로 +2
        
        return change_points
    
    def detect_direction_changes(self, actions: np.ndarray) -> List[int]:
        """
        관절 이동 방향 반전 시점 감지
        
        일정 윈도우 내 이동 방향 벡터의 코사인 유사도가
        급격히 떨어지는(반전되는) 시점을 감지합니다.
        
        Args:
            actions: [num_frames, action_dim]
            
        Returns:
            change_points: 방향이 반전되는 프레임 인덱스 리스트
        """
        joint_actions = actions[:, :-1]
        
        # 윈도우 기반 방향 벡터 계산
        window = 10
        change_points = []
        
        for i in range(window, len(joint_actions) - window):
            # 이전 윈도우의 이동 방향
            dir_before = joint_actions[i] - joint_actions[i - window]
            # 이후 윈도우의 이동 방향
            dir_after = joint_actions[i + window] - joint_actions[i]
            
            # 코사인 유사도
            norm_before = np.linalg.norm(dir_before) + 1e-8
            norm_after = np.linalg.norm(dir_after) + 1e-8
            
            cosine_sim = np.dot(dir_before, dir_after) / (norm_before * norm_after)
            
            # 코사인 유사도가 낮으면 (방향 반전) → 분할점
            if cosine_sim < -self.direction_change_threshold:
                change_points.append(i)
        
        return change_points
    
    def merge_close_points(
        self,
        points: List[int],
        min_distance: int,
    ) -> List[int]:
        """
        너무 가까운 분할점 병합
        
        동일 이벤트에서 여러 신호가 동시에 발생하면
        분할점이 중복될 수 있으므로 병합합니다.
        """
        if not points:
            return []
        
        sorted_points = sorted(set(points))
        merged = [sorted_points[0]]
        
        for p in sorted_points[1:]:
            if p - merged[-1] >= min_distance:
                merged.append(p)
        
        return merged
    
    def determine_gripper_state(self, gripper_values: np.ndarray) -> str:
        """세그먼트 내 그리퍼 상태 판단"""
        mean_val = np.mean(gripper_values)
        std_val = np.std(gripper_values)
        
        if std_val > 0.2:
            return "transitioning"
        elif mean_val > 0.5:
            return "closed"
        else:
            return "open"
    
    def determine_dominant_direction(
        self,
        actions: np.ndarray,
    ) -> str:
        """
        세그먼트 내 지배적 이동 방향 판단
        
        SO-100 관절 구조 기반:
        - shoulder_pan (0): 좌우 회전
        - shoulder_lift (1): 상하 이동 (주요)
        - elbow_flex (2): 전후 이동 (주요)
        - wrist_flex (3): 미세 조정
        - wrist_roll (4): 회전
        """
        if len(actions) < 2:
            return "stationary"
        
        displacement = actions[-1, :-1] - actions[0, :-1]  # gripper 제외
        
        total_disp = np.linalg.norm(displacement)
        if total_disp < 0.05:
            return "stationary"
        
        # 가장 큰 변위를 가진 관절 기준
        max_joint = np.argmax(np.abs(displacement))
        max_disp = displacement[max_joint]
        
        direction_map = {
            0: ("left", "right"),
            1: ("up", "down"),
            2: ("forward", "backward"),
            3: ("wrist_up", "wrist_down"),
            4: ("rotate_cw", "rotate_ccw"),
        }
        
        if max_joint in direction_map:
            pos_dir, neg_dir = direction_map[max_joint]
            return pos_dir if max_disp > 0 else neg_dir
        
        return "mixed"
    
    def segment_episode(
        self,
        actions: np.ndarray,
        episode_id: int = 0,
        task_description: str = "",
        dataset_name: str = "",
        fps: int = 30,
    ) -> EpisodeSegmentation:
        """
        단일 에피소드를 세그먼트로 분할
        
        Args:
            actions: [num_frames, action_dim] 전체 에피소드 action 데이터
            episode_id: 에피소드 ID
            task_description: 태스크 설명 (데이터셋에서 가져옴)
            dataset_name: 데이터셋 이름
            fps: 프레임 레이트
            
        Returns:
            EpisodeSegmentation 결과
        """
        num_frames = len(actions)
        
        # === 3가지 신호로 분할점 감지 ===
        gripper_points = self.detect_gripper_changes(actions)
        velocity_points = self.detect_velocity_changes(actions)
        direction_points = self.detect_direction_changes(actions)
        
        # 분할점에 트리거 정보 태깅
        all_points_with_trigger = []
        for p in gripper_points:
            all_points_with_trigger.append((p, "gripper"))
        for p in velocity_points:
            all_points_with_trigger.append((p, "velocity"))
        for p in direction_points:
            all_points_with_trigger.append((p, "direction"))
        
        # 프레임 순서로 정렬
        all_points_with_trigger.sort(key=lambda x: x[0])
        
        # 가까운 점 병합 (min_segment_frames 기준)
        merged_points = []
        merged_triggers = []
        
        if all_points_with_trigger:
            merged_points.append(all_points_with_trigger[0][0])
            merged_triggers.append(all_points_with_trigger[0][1])
            
            for point, trigger in all_points_with_trigger[1:]:
                if point - merged_points[-1] >= self.min_segment_frames:
                    merged_points.append(point)
                    merged_triggers.append(trigger)
        
        # 경계 추가 (에피소드 시작/끝)
        boundaries = [0] + merged_points + [num_frames]
        triggers = ["start"] + merged_triggers
        
        # === 세그먼트 생성 ===
        segments = []
        seg_id = 0
        
        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i + 1]
            duration = end - start
            
            # 너무 짧은 세그먼트는 건너뜀
            if duration < self.min_segment_frames and i > 0 and i < len(boundaries) - 2:
                continue
            
            # 대표 프레임: 세그먼트 중간 지점
            representative = start + duration // 2
            
            # 세그먼트 내 action 데이터 추출
            seg_actions = actions[start:end]
            
            segment = Segment(
                segment_id=seg_id,
                start_frame=int(start),
                end_frame=int(end),
                representative_frame=int(representative),
                trigger=triggers[i] if i < len(triggers) else "end",
                duration_frames=int(duration),
                gripper_state=self.determine_gripper_state(seg_actions[:, -1]),
                mean_velocity=float(np.mean(np.linalg.norm(np.diff(seg_actions[:, :-1], axis=0), axis=1)))
                              if len(seg_actions) > 1 else 0.0,
                dominant_direction=self.determine_dominant_direction(seg_actions),
            )
            
            segments.append(asdict(segment))
            seg_id += 1
        
        return EpisodeSegmentation(
            episode_id=episode_id,
            task_description=task_description,
            total_frames=num_frames,
            segments=segments,
            dataset_name=dataset_name,
            fps=fps,
        )
    
    def process_dataset(
        self,
        dataset_repo_id: str,
        dataset_root: Optional[str] = None,
        max_episodes: Optional[int] = None,
    ) -> List[dict]:
        """
        전체 데이터셋 처리
        
        LeRobot 데이터셋을 로드하고 모든 에피소드를 분할합니다.
        
        Args:
            dataset_repo_id: LeRobot dataset ID (예: "community_dataset_v1")
            dataset_root: 로컬 데이터 경로
            max_episodes: 최대 처리 에피소드 수
            
        Returns:
            모든 에피소드의 세그먼트 정보 리스트
        """
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
            
            print(f"  Loading dataset: {dataset_repo_id}")
            if dataset_root:
                dataset = LeRobotDataset(dataset_repo_id, root=dataset_root)
            else:
                dataset = LeRobotDataset(dataset_repo_id)
            
            num_episodes = len(dataset.meta.episodes)
            if max_episodes:
                num_episodes = min(num_episodes, max_episodes)
            
            print(f"  Processing {num_episodes} episodes...")
            
            all_results = []
            
            for ep_idx in range(num_episodes):
                if ep_idx % 500 == 0:
                    print(f"    Episode {ep_idx}/{num_episodes}")
                
                # 에피소드 프레임 범위
                from_idx = dataset.meta.episodes["dataset_from_index"][ep_idx]
                to_idx = dataset.meta.episodes["dataset_to_index"][ep_idx]
                
                # Action 데이터 추출
                actions = []
                for frame_idx in range(from_idx, to_idx):
                    item = dataset[frame_idx]
                    action = item["action"].numpy()
                    actions.append(action)
                
                if not actions:
                    continue
                
                actions_array = np.stack(actions)
                
                # Task description 추출
                task_desc = ""
                sample = dataset[from_idx]
                if "task" in sample:
                    task_desc = sample["task"]
                elif "language_instruction" in sample:
                    task_desc = sample["language_instruction"]
                
                # 세그먼트 분할
                result = self.segment_episode(
                    actions=actions_array,
                    episode_id=ep_idx,
                    task_description=task_desc,
                    dataset_name=dataset_repo_id,
                    fps=dataset.meta.fps if hasattr(dataset.meta, 'fps') else 30,
                )
                
                all_results.append(asdict(result))
            
            return all_results
            
        except ImportError:
            print("  [WARNING] lerobot not installed. Using dummy data for testing.")
            return self._generate_dummy_data(max_episodes or 10)
    
    def _generate_dummy_data(self, num_episodes: int = 10) -> List[dict]:
        """테스트용 더미 데이터 생성"""
        all_results = []
        
        for ep_idx in range(num_episodes):
            # 랜덤 에피소드 생성 (300~600 frames)
            num_frames = np.random.randint(300, 600)
            actions = np.random.randn(num_frames, 7) * 0.1
            
            # 그리퍼 상태 시뮬레이션 (중간에 2번 변화)
            gripper = np.zeros(num_frames)
            change1 = num_frames // 3
            change2 = 2 * num_frames // 3
            gripper[change1:change2] = 1.0
            actions[:, -1] = gripper
            
            result = self.segment_episode(
                actions=actions,
                episode_id=ep_idx,
                task_description=f"Pick and place object {ep_idx}",
                dataset_name="dummy_dataset",
                fps=30,
            )
            
            all_results.append(asdict(result))
        
        return all_results


if __name__ == "__main__":
    print("Testing AutoSegmenter with dummy data...")
    
    segmenter = AutoSegmenter()
    
    # 더미 에피소드 생성 및 테스트
    np.random.seed(42)
    num_frames = 450  # 15초 × 30fps
    
    # 현실적인 pick-and-place 시뮬레이션
    actions = np.zeros((num_frames, 7))  # 6 joints + gripper
    
    # Phase 1: Approach (frame 0~100) - 앞으로 이동
    for i in range(100):
        actions[i, 2] = i * 0.01   # elbow forward
        actions[i, 1] = -i * 0.005  # shoulder down
    
    # Phase 2: Grasp (frame 100~150) - 그리퍼 닫기
    actions[100:150, 2] = 1.0
    actions[100:150, 1] = -0.5
    for i in range(50):
        actions[100 + i, -1] = i / 50.0  # gripper closing
    
    # Phase 3: Lift (frame 150~250) - 위로 이동
    actions[150:250, -1] = 1.0  # gripper closed
    for i in range(100):
        actions[150 + i, 1] = -0.5 + i * 0.01  # shoulder up
        actions[150 + i, 2] = 1.0
    
    # Phase 4: Move (frame 250~350) - 옆으로 이동
    actions[250:350, -1] = 1.0
    actions[250:350, 1] = 0.5
    for i in range(100):
        actions[250 + i, 0] = i * 0.01  # shoulder pan
        actions[250 + i, 2] = 1.0
    
    # Phase 5: Place (frame 350~450) - 내려놓기
    actions[350:450, 0] = 1.0
    for i in range(50):
        actions[350 + i, 1] = 0.5 - i * 0.01
    for i in range(50):
        actions[400 + i, -1] = 1.0 - i / 50.0  # gripper opening
    
    # 약간의 노이즈 추가
    actions += np.random.randn(*actions.shape) * 0.01
    
    result = segmenter.segment_episode(
        actions=actions,
        episode_id=0,
        task_description="Pick up object and place it to the right",
        dataset_name="test",
        fps=30,
    )
    
    print(f"\nEpisode: {result.task_description}")
    print(f"Total frames: {result.total_frames}")
    print(f"Segments: {len(result.segments)}")
    
    for seg in result.segments:
        print(f"  [{seg['segment_id']}] frame {seg['start_frame']:3d}-{seg['end_frame']:3d} "
              f"| trigger={seg['trigger']:10s} "
              f"| gripper={seg['gripper_state']:13s} "
              f"| dir={seg['dominant_direction']:10s} "
              f"| repr_frame={seg['representative_frame']}")
    
    print("\n✓ AutoSegmenter test completed!")