"""
Stage 1: Rule-based Auto Segmentation
======================================

로봇 action 데이터를 분석하여 에피소드를 의미 있는 세그먼트로 자동 분할합니다.

분할 기준 (3가지 신호의 OR 조합):
  1. Gripper state change
  2. Velocity change
  3. Direction change
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import json


@dataclass
class Segment:
    segment_id: int
    start_frame: int
    end_frame: int
    representative_frame: int
    trigger: str
    duration_frames: int
    gripper_state: str
    mean_velocity: float
    dominant_direction: str


@dataclass
class EpisodeSegmentation:
    episode_id: int
    task_description: str
    total_frames: int
    segments: List[dict]
    dataset_name: str
    fps: int


class AutoSegmenter:
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
        if len(actions) < 2:
            return []
        gripper = actions[:, -1]
        gripper_diff = np.abs(np.diff(gripper))
        return [i + 1 for i in range(len(gripper_diff)) if gripper_diff[i] > self.gripper_threshold]

    def detect_velocity_changes(self, actions: np.ndarray) -> List[int]:
        # 프레임 수 안전 가드
        if len(actions) < 4:
            return []

        joint_actions = actions[:, :-1] if actions.shape[-1] > 1 else actions
        velocities = np.diff(joint_actions, axis=0)
        speed = np.linalg.norm(velocities, axis=1)

        if len(speed) == 0:
            return []

        window = 5
        if len(speed) > window:
            kernel = np.ones(window) / window
            speed_smooth = np.convolve(speed, kernel, mode='same')
        else:
            speed_smooth = speed.copy()

        if len(speed_smooth) == 0:
            return []

        speed_diff = np.abs(np.diff(speed_smooth))
        if len(speed_diff) == 0:
            return []

        speed_range = float(speed_smooth.max()) - float(speed_smooth.min())
        adaptive_threshold = self.velocity_threshold * (speed_range + 1e-6)

        return [i + 2 for i in range(len(speed_diff)) if speed_diff[i] > adaptive_threshold]

    def detect_direction_changes(self, actions: np.ndarray) -> List[int]:
        if len(actions) < 3:
            return []

        joint_actions = actions[:, :-1] if actions.shape[-1] > 1 else actions
        window = min(10, len(joint_actions) // 3)
        if window < 1:
            return []

        change_points = []
        for i in range(window, len(joint_actions) - window):
            dir_before = joint_actions[i] - joint_actions[i - window]
            dir_after  = joint_actions[i + window] - joint_actions[i]
            norm_before = np.linalg.norm(dir_before) + 1e-8
            norm_after  = np.linalg.norm(dir_after)  + 1e-8
            cosine_sim = np.dot(dir_before, dir_after) / (norm_before * norm_after)
            if cosine_sim < -self.direction_change_threshold:
                change_points.append(i)
        return change_points

    def determine_gripper_state(self, gripper_values: np.ndarray) -> str:
        if len(gripper_values) == 0:
            return "unknown"
        mean_val = float(np.mean(gripper_values))
        std_val  = float(np.std(gripper_values))
        if std_val > 0.2:
            return "transitioning"
        return "closed" if mean_val > 0.5 else "open"

    def determine_dominant_direction(self, actions: np.ndarray) -> str:
        if len(actions) < 2:
            return "stationary"
        displacement = actions[-1, :-1] - actions[0, :-1] if actions.shape[-1] > 1 else actions[-1] - actions[0]
        total_disp = float(np.linalg.norm(displacement))
        if total_disp < 0.05:
            return "stationary"
        max_joint = int(np.argmax(np.abs(displacement)))
        max_disp  = float(displacement[max_joint])
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
        num_frames = len(actions)

        # 에피소드가 너무 짧으면 단일 세그먼트
        if num_frames < self.min_segment_frames * 2:
            seg = Segment(
                segment_id=0,
                start_frame=0,
                end_frame=num_frames,
                representative_frame=num_frames // 2,
                trigger="start",
                duration_frames=num_frames,
                gripper_state=self.determine_gripper_state(actions[:, -1]) if actions.shape[-1] > 0 else "unknown",
                mean_velocity=0.0,
                dominant_direction=self.determine_dominant_direction(actions),
            )
            return EpisodeSegmentation(
                episode_id=episode_id,
                task_description=task_description,
                total_frames=num_frames,
                segments=[asdict(seg)],
                dataset_name=dataset_name,
                fps=fps,
            )

        gripper_points   = self.detect_gripper_changes(actions)
        velocity_points  = self.detect_velocity_changes(actions)
        direction_points = self.detect_direction_changes(actions)

        all_points_with_trigger = (
            [(p, "gripper")   for p in gripper_points]  +
            [(p, "velocity")  for p in velocity_points]  +
            [(p, "direction") for p in direction_points]
        )
        all_points_with_trigger.sort(key=lambda x: x[0])

        merged_points   = []
        merged_triggers = []

        if all_points_with_trigger:
            merged_points.append(all_points_with_trigger[0][0])
            merged_triggers.append(all_points_with_trigger[0][1])
            for point, trigger in all_points_with_trigger[1:]:
                if point - merged_points[-1] >= self.min_segment_frames:
                    merged_points.append(point)
                    merged_triggers.append(trigger)

        boundaries = [0] + merged_points + [num_frames]
        triggers   = ["start"] + merged_triggers

        segments = []
        seg_id   = 0

        for i in range(len(boundaries) - 1):
            start    = boundaries[i]
            end      = boundaries[i + 1]
            duration = end - start

            if duration < self.min_segment_frames and 0 < i < len(boundaries) - 2:
                continue

            representative = start + duration // 2
            seg_actions    = actions[start:end]

            gripper_col = seg_actions[:, -1] if seg_actions.shape[-1] > 0 else np.array([0.0])
            if len(seg_actions) > 1:
                vels = np.diff(seg_actions[:, :-1] if seg_actions.shape[-1] > 1 else seg_actions, axis=0)
                mean_vel = float(np.mean(np.linalg.norm(vels, axis=1))) if len(vels) > 0 else 0.0
            else:
                mean_vel = 0.0

            seg = Segment(
                segment_id=seg_id,
                start_frame=int(start),
                end_frame=int(end),
                representative_frame=int(representative),
                trigger=triggers[i] if i < len(triggers) else "end",
                duration_frames=int(duration),
                gripper_state=self.determine_gripper_state(gripper_col),
                mean_velocity=mean_vel,
                dominant_direction=self.determine_dominant_direction(seg_actions),
            )
            segments.append(asdict(seg))
            seg_id += 1

        if not segments:
            seg = Segment(
                segment_id=0,
                start_frame=0,
                end_frame=num_frames,
                representative_frame=num_frames // 2,
                trigger="start",
                duration_frames=num_frames,
                gripper_state="unknown",
                mean_velocity=0.0,
                dominant_direction="mixed",
            )
            segments = [asdict(seg)]

        return EpisodeSegmentation(
            episode_id=episode_id,
            task_description=task_description,
            total_frames=num_frames,
            segments=segments,
            dataset_name=dataset_name,
            fps=fps,
        )

    def process_dataset(self, dataset_repo_id, dataset_root=None, max_episodes=None):
        return []


if __name__ == "__main__":
    print("Testing AutoSegmenter...")
    segmenter = AutoSegmenter()
    np.random.seed(42)
    actions = np.zeros((450, 7))
    for i in range(100):
        actions[i, 2] = i * 0.01
    actions[100:150, -1] = np.linspace(0, 1, 50)
    actions[150:250, -1] = 1.0
    result = segmenter.segment_episode(actions, 0, "Test task", "test", 30)
    print(f"Segments: {len(result.segments)}")
    for s in result.segments:
        print(f"  [{s['segment_id']}] {s['start_frame']}-{s['end_frame']} trigger={s['trigger']}")
    print("✓ Done")