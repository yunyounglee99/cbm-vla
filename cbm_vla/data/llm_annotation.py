"""
Stage 2: Gemini API Concept Annotation
========================================

각 세그먼트의 대표 프레임 이미지를 Gemini에 보내
행동 컨셉(action concept) 문장을 생성합니다.

=== Prompt 설계 원칙 ===

1. System Role: 로봇 manipulation 전문 annotator로 설정
2. Image Context: 대표 프레임 1장 (LOW resolution으로 토큰 절약)
3. Text Context: 에피소드의 task description + 세그먼트 메타데이터
4. Output Format: 정해진 JSON 구조로 강제

=== Prompt가 반드시 포함해야 하는 정보 ===

[Image에서 읽어야 하는 것]
  - 로봇 팔의 현재 자세 (접근 중? 잡고 있는 중? 들고 있는 중?)
  - 대상 물체의 위치와 상태
  - 배경/환경 정보 (테이블 위 물건 배치 등)

[Text에서 제공하는 것]
  - 전체 task instruction ("Pick up the red cube and place in box")
  - 현재 세그먼트의 시간적 위치 (에피소드 시작/중간/끝)
  - 그리퍼 상태 (open/closed/transitioning)
  - 지배적 이동 방향 (forward/up/lateral 등)
  - 평균 속도 (빠른 이동 vs 정밀 조작 구분)

[Gemini가 생성해야 하는 것]
  - action_concept: 짧은 행동 컨셉명 (예: "approach_object")
  - description: 구체적 행동 설명 (1문장)
  - image_description: 이미지 내 장면 설명 (CBM encoder 학습용)
"""

import base64
import json
import time
import io
import os
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass


# ================================================================
# Prompt Templates
# ================================================================

SYSTEM_PROMPT = """You are an expert robotic manipulation annotator. Your task is to analyze images from a robotic arm performing tabletop manipulation tasks and generate precise action concept labels.

The robot is a SO-100 6-DOF follower arm with a gripper, performing tasks on a table surface. You will see one frame from a specific segment of the robot's task execution.

RULES:
1. The action_concept must be a short, standardized verb phrase (2-4 words, snake_case)
2. Use ONLY from this vocabulary when possible: approach, reach, align, grasp, grip, hold, lift, raise, lower, move, transport, carry, place, release, drop, push, pull, rotate, flip, insert, pour, scoop, wipe, retract, hover, stabilize, adjust
3. The description must be ONE concrete sentence describing what the robot is doing
4. The image_description must describe the visual scene objectively (objects, positions, robot pose)
5. Respond ONLY with the JSON object, no markdown, no extra text"""

USER_PROMPT_TEMPLATE = """Analyze this robotic manipulation frame and generate an action concept.

TASK: {task_description}

SEGMENT CONTEXT:
- Position in episode: segment {segment_id} of {total_segments} (frame {start_frame}-{end_frame} of {total_frames})
- Gripper state: {gripper_state}
- Dominant movement direction: {dominant_direction}
- Average joint velocity: {mean_velocity:.4f} ({"fast movement" if {mean_velocity} > 0.05 else "slow/precise movement" if {mean_velocity} > 0.01 else "nearly stationary"})
- Trigger for this segment: {trigger}

Based on the image and context above, respond with this exact JSON format:
{{
  "action_concept": "verb_phrase_in_snake_case",
  "description": "One sentence describing the robot's current action",
  "image_description": "One sentence describing the visual scene"
}}"""

# 속도 기반 텍스트를 미리 생성하기 위한 헬퍼
def _velocity_text(v: float) -> str:
    if v > 0.05:
        return "fast movement"
    elif v > 0.01:
        return "slow/precise movement"
    else:
        return "nearly stationary"


def build_user_prompt(segment: dict, episode_info: dict) -> str:
    """
    세그먼트 정보로부터 user prompt를 생성합니다.
    
    === Prompt 설계 상세 설명 ===
    
    1. TASK 줄: 전체 에피소드의 목표를 제공하여 Gemini가 맥락을 이해하게 함
       → "이 로봇은 빨간 큐브를 집어서 상자에 넣으려고 한다"를 알면
       → 현재 프레임이 "approach" 단계인지 "grasp" 단계인지 판단 가능
    
    2. SEGMENT CONTEXT: 세그먼트의 메타데이터를 텍스트로 제공
       - Position: "5개 중 2번째 세그먼트" → 시간적 맥락
       - Gripper state: "closed" → 이미 물체를 잡았다는 것을 암시
       - Direction: "up" → 들어올리는 동작일 가능성
       - Velocity: "slow" → 정밀 조작 (잡기, 놓기 등)
       - Trigger: "gripper" → 그리퍼 상태 변화가 이 세그먼트를 만들었음
       
       이 메타데이터가 중요한 이유: 이미지 한 장만으로는 로봇의 "의도"를
       판단하기 어렵습니다. 예를 들어 같은 자세라도 "내려놓기 직전"인지
       "집기 직전"인지는 맥락 없이는 구분 불가. 메타데이터가 이를 보완합니다.
    
    3. JSON Format 강제: Gemini가 정확히 3개 필드만 출력하도록 합니다.
       - action_concept: scoring module의 concept pool에 들어갈 이름
       - description: 더 풍부한 텍스트 (T5 refinement에서 클러스터링에 활용)
       - image_description: CBM encoder 학습용 (이미지-텍스트 대응)
    """
    mv = segment.get("mean_velocity", 0.0)
    velocity_desc = _velocity_text(mv)
    
    total_segs = len(episode_info.get("segments", []))
    
    robot_type = episode_info.get("robot_type", "unknown_robot")
    embodiment_task = f"[Robot: {robot_type}] {episode_info.get('task_description', 'Unknown task')}"

    prompt = f"""Analyze this robotic manipulation frame and generate an action concept.

TASK: {embodiment_task}

SEGMENT CONTEXT:
- Position in episode: segment {segment['segment_id']} of {total_segs} (frame {segment['start_frame']}-{segment['end_frame']} of {episode_info.get('total_frames', 0)})
- Gripper state: {segment.get('gripper_state', 'unknown')}
- Dominant movement direction: {segment.get('dominant_direction', 'unknown')}
- Average joint velocity: {mv:.4f} ({velocity_desc})
- Trigger for this segment: {segment.get('trigger', 'unknown')}

Based on the image and context above, respond with this exact JSON format:
{{
  "action_concept": "verb_phrase_in_snake_case",
  "description": "One sentence describing the robot's current action",
  "image_description": "One sentence describing the visual scene"
}}"""
    
    return prompt


# ================================================================
# Gemini API Client
# ================================================================

class GeminiConceptAnnotator:
    """
    Gemini API를 사용한 세그먼트별 컨셉 생성
    
    Args:
        api_key: Gemini API 키
        model_name: 사용할 모델 (기본: gemini-2.0-flash)
        max_rpm: 분당 최대 요청 수 (rate limiting)
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "google/gemini-2.5-flash", # OpenRouter 모델명으로 변경
        max_rpm: int = 60,
    ):
        # OpenRouter API 키를 환경변수에서 가져오도록 수정
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY") 
        self.model_name = model_name
        self.max_rpm = max_rpm
        self._request_interval = 60.0 / max_rpm
        self._last_request_time = 0
        
        self._client = None
    
    def _init_client(self):
        """OpenRouter 호환 OpenAI 클라이언트 초기화"""
        if self._client is not None:
            return
        
        try:
            from openai import OpenAI
            self._client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=self.api_key,
            )
            print(f"  OpenRouter client initialized: {self.model_name}")
        except ImportError:
            print("  [WARNING] openai not installed.")
            print("  Install: pip install openai")
            self._client = None
    
    def _rate_limit(self):
        """Rate limiting 적용"""
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < self._request_interval:
            time.sleep(self._request_interval - elapsed)
        self._last_request_time = time.time()
    
    def _get_base64_image(self, image_path: str) -> Optional[str]:
        """이미지를 OpenRouter용 Base64 문자열로 로드 및 변환"""
        try:
            import PIL.Image
            img = PIL.Image.open(image_path)
            # LOW resolution으로 리사이즈 (토큰 절약: 320x240)
            img = img.resize((320, 240), PIL.Image.LANCZOS)
            
            # 이미지를 메모리 버퍼에 JPEG로 저장 후 Base64 인코딩
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            return img_str
        except Exception as e:
            print(f"    [ERROR] Failed to load/encode image {image_path}: {e}")
            return None

    def annotate_single(
        self,
        image_path: str,
        segment: dict,
        episode_info: dict,
    ) -> Optional[dict]:
        self._init_client()
        
        if self._client is None:
            return self._fallback_annotation(segment, episode_info)
        
        # Base64 이미지 변환
        base64_image = self._get_base64_image(image_path)
        if base64_image is None:
            return self._fallback_annotation(segment, episode_info)
        
        # 프롬프트 생성
        prompt = build_user_prompt(segment, episode_info)
        
        # Rate limiting
        self._rate_limit()
        
        # API 호출 (OpenRouter / OpenAI 구조)
        try:
            response = self._client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                response_format={"type": "json_object"}, # JSON 강제 출력
                temperature=0.2,
                top_p=0.8,
                max_tokens=256,
            )
            
            # JSON 파싱
            response_content = response.choices[0].message.content
            result = json.loads(response_content)
            
            # 필드 검증 및 포맷팅 (기존 로직 유지)
            required_fields = ["action_concept", "description", "image_description"]
            for field in required_fields:
                if field not in result:
                    result[field] = "unknown"
            
            result["action_concept"] = (
                result["action_concept"]
                .lower()
                .strip()
                .replace(" ", "_")
                .replace("-", "_")
            )
            
            return result
            
        except Exception as e:
            print(f"    [ERROR] OpenRouter API call failed: {e}")
            return self._fallback_annotation(segment, episode_info)
    
    def _extract_frame_image(
        self,
        dataset_root: str,
        episode_id: int,
        frame_idx: int,
        camera: str = "observation.images.top",
    ) -> Optional[str]:
        """
        LeRobot 데이터셋에서 특정 프레임의 이미지 경로를 추출
        
        LeRobot v2 데이터 구조:
          dataset_root/videos/{camera}_episode_{ep_id:06d}.mp4
          또는 dataset_root/data/{camera}/frame_{frame_idx:06d}.png
        """
        root = Path(dataset_root) if dataset_root else None
        if root is None:
            return None
        
        # 방법 1: 개별 프레임 파일
        camera_dir = camera.replace(".", "/")
        frame_path = root / "data" / camera_dir / f"frame_{frame_idx:06d}.png"
        if frame_path.exists():
            return str(frame_path)
        
        # 방법 2: 비디오에서 프레임 추출
        video_path = root / "videos" / f"{camera}_episode_{episode_id:06d}.mp4"
        if video_path.exists():
            return self._extract_frame_from_video(str(video_path), frame_idx)
        
        return None
    
    def _extract_frame_from_video(
        self,
        video_path: str,
        frame_idx: int,
    ) -> Optional[str]:
        """비디오에서 특정 프레임을 추출하여 임시 파일로 저장"""
        try:
            import cv2
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                tmp_path = f"/tmp/cbmvla_frame_{frame_idx}.jpg"
                cv2.imwrite(tmp_path, frame)
                return tmp_path
        except ImportError:
            pass
        
        return None
    
    def _fallback_annotation(
        self,
        segment: dict,
        episode_info: dict,
    ) -> dict:
        """
        API 없이 메타데이터 기반으로 규칙적 컨셉 생성 (테스트/fallback용)
        
        이것은 Gemini API가 없는 환경에서 파이프라인 테스트를 위한 것입니다.
        실제 사용 시에는 반드시 Gemini API를 사용해야 합니다.
        """
        gripper = segment.get("gripper_state", "unknown")
        direction = segment.get("dominant_direction", "unknown")
        velocity = segment.get("mean_velocity", 0)
        trigger = segment.get("trigger", "unknown")
        seg_id = segment.get("segment_id", 0)
        total_segs = len(episode_info.get("segments", []))
        
        # 규칙 기반 컨셉 추론
        if seg_id == 0 and total_segs > 1:
            concept = "approach_object"
            desc = "Robot arm approaches the target object"
        elif trigger == "gripper" and gripper == "transitioning":
            if seg_id < total_segs // 2:
                concept = "grasp_object"
                desc = "Robot gripper closes to grasp the object"
            else:
                concept = "release_object"
                desc = "Robot gripper opens to release the object"
        elif gripper == "closed" and direction in ("up", "raise"):
            concept = "lift_object"
            desc = "Robot lifts the grasped object upward"
        elif gripper == "closed" and direction in ("left", "right", "forward", "backward"):
            concept = "transport_object"
            desc = f"Robot moves the object {direction}"
        elif gripper == "closed" and direction in ("down", "lower"):
            concept = "lower_object"
            desc = "Robot lowers the object toward the surface"
        elif velocity < 0.005:
            concept = "stabilize"
            desc = "Robot holds position to stabilize"
        elif seg_id == total_segs - 1:
            concept = "retract"
            desc = "Robot retracts after completing the task"
        else:
            concept = f"move_{direction}" if direction != "unknown" else "move_arm"
            desc = f"Robot moves arm {direction}"
        
        task = episode_info.get("task_description", "manipulation task")
        
        return {
            "action_concept": concept,
            "description": desc,
            "image_description": f"Robotic arm performing {task} on a tabletop surface",
        }
    
    def annotate_segments(
        self,
        segments: List[dict],
        dataset_root: Optional[str] = None,
        output_path: Optional[str] = None,
        batch_size: int = 10,
    ) -> List[dict]:
        """
        모든 세그먼트에 대한 컨셉 생성
        
        Args:
            segments: AutoSegmenter 출력 (에피소드별 세그먼트 리스트)
            dataset_root: 이미지를 가져올 데이터셋 경로
            output_path: 중간 결과 저장 경로 (재시작용)
            batch_size: 로깅 간격
            
        Returns:
            세그먼트별 컨셉 annotation 리스트
        """
        all_annotations = []
        
        # 기존 결과 로드 (재시작 지원)
        processed_keys = set()
        if output_path and Path(output_path).exists():
            with open(output_path) as f:
                all_annotations = json.load(f)
            processed_keys = {
                (a["episode_id"], a["segment_id"]) for a in all_annotations
            }
            print(f"  Loaded {len(all_annotations)} existing annotations")
        
        total_to_process = sum(len(ep["segments"]) for ep in segments)
        processed = len(processed_keys)
        
        print(f"  Total segments to annotate: {total_to_process}")
        print(f"  Already processed: {processed}")
        print(f"  Remaining: {total_to_process - processed}")
        
        for ep in segments:
            episode_id = ep["episode_id"]
            
            for seg in ep["segments"]:
                seg_id = seg["segment_id"]
                
                # 이미 처리된 세그먼트 건너뜀
                if (episode_id, seg_id) in processed_keys:
                    continue
                
                # 대표 프레임 이미지 경로
                image_path = None
                if dataset_root:
                    image_path = self._extract_frame_image(
                        dataset_root=dataset_root,
                        episode_id=episode_id,
                        frame_idx=seg["representative_frame"],
                    )
                
                # 컨셉 생성
                if image_path:
                    annotation = self.annotate_single(image_path, seg, ep)
                else:
                    annotation = self._fallback_annotation(seg, ep)
                
                # 메타데이터 추가
                annotation["episode_id"] = episode_id
                annotation["segment_id"] = seg_id
                annotation["start_frame"] = seg["start_frame"]
                annotation["end_frame"] = seg["end_frame"]
                annotation["representative_frame"] = seg["representative_frame"]
                annotation["task_description"] = ep.get("task_description", "")
                
                all_annotations.append(annotation)
                processed += 1
                
                # 주기적 저장 및 로깅
                if processed % batch_size == 0:
                    print(f"    Processed {processed}/{total_to_process} segments")
                    if output_path:
                        with open(output_path, "w") as f:
                            json.dump(all_annotations, f, indent=2, ensure_ascii=False)
        
        # 최종 저장
        if output_path:
            with open(output_path, "w") as f:
                json.dump(all_annotations, f, indent=2, ensure_ascii=False)
        
        print(f"  Annotation complete: {len(all_annotations)} segments")
        return all_annotations
    
    def estimate_cost(
        self,
        num_segments: int,
        model: str = "gemini-2.0-flash",
    ) -> dict:
        """API 비용 예상"""
        # 세그먼트당 토큰 수 (이미지 + 텍스트)
        input_tokens_per_seg = 1100   # image ~800 + prompt ~300
        output_tokens_per_seg = 150   # JSON 출력
        
        total_input = num_segments * input_tokens_per_seg
        total_output = num_segments * output_tokens_per_seg
        
        pricing = {
            "gemini-2.0-flash":      {"input": 0.10, "output": 0.40},
            "gemini-2.0-flash-lite": {"input": 0.075, "output": 0.30},
            "gemini-2.5-flash":      {"input": 0.30, "output": 2.50},
        }
        
        prices = pricing.get(model, pricing["gemini-2.0-flash"])
        
        input_cost = (total_input / 1_000_000) * prices["input"]
        output_cost = (total_output / 1_000_000) * prices["output"]
        
        return {
            "model": model,
            "num_segments": num_segments,
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "input_cost_usd": round(input_cost, 2),
            "output_cost_usd": round(output_cost, 2),
            "total_cost_usd": round(input_cost + output_cost, 2),
            "batch_discount_50pct": round((input_cost + output_cost) * 0.5, 2),
        }


if __name__ == "__main__":
    print("Testing GeminiConceptAnnotator...")
    
    annotator = GeminiConceptAnnotator()
    
    # 비용 예상 테스트
    print("\n=== Cost Estimation ===")
    for model in ["gemini-2.0-flash", "gemini-2.5-flash", "gemini-2.0-flash-lite"]:
        for n_eps in [11100, 23000]:
            n_segs = int(n_eps * 4.5)
            cost = annotator.estimate_cost(n_segs, model)
            print(f"  {model} | {n_eps:,} episodes ({n_segs:,} segments)")
            print(f"    Input: {cost['total_input_tokens']:,} tokens → ${cost['input_cost_usd']}")
            print(f"    Output: {cost['total_output_tokens']:,} tokens → ${cost['output_cost_usd']}")
            print(f"    Total: ${cost['total_cost_usd']} (batch 50%: ${cost['batch_discount_50pct']})")
            print()
    
    # Fallback annotation 테스트
    print("\n=== Fallback Annotation Test ===")
    dummy_episode = {
        "episode_id": 0,
        "task_description": "Pick up the red cube and place it in the box",
        "total_frames": 450,
        "segments": [
            {"segment_id": 0, "start_frame": 0, "end_frame": 100,
             "gripper_state": "open", "dominant_direction": "forward",
             "mean_velocity": 0.03, "trigger": "start", "representative_frame": 50},
            {"segment_id": 1, "start_frame": 100, "end_frame": 150,
             "gripper_state": "transitioning", "dominant_direction": "stationary",
             "mean_velocity": 0.005, "trigger": "gripper", "representative_frame": 125},
            {"segment_id": 2, "start_frame": 150, "end_frame": 250,
             "gripper_state": "closed", "dominant_direction": "up",
             "mean_velocity": 0.04, "trigger": "velocity", "representative_frame": 200},
            {"segment_id": 3, "start_frame": 250, "end_frame": 350,
             "gripper_state": "closed", "dominant_direction": "right",
             "mean_velocity": 0.06, "trigger": "direction", "representative_frame": 300},
            {"segment_id": 4, "start_frame": 350, "end_frame": 450,
             "gripper_state": "transitioning", "dominant_direction": "down",
             "mean_velocity": 0.02, "trigger": "gripper", "representative_frame": 400},
        ],
    }
    
    for seg in dummy_episode["segments"]:
        result = annotator._fallback_annotation(seg, dummy_episode)
        print(f"  Seg {seg['segment_id']}: {result['action_concept']:20s} → {result['description']}")
    
    # Prompt 생성 테스트
    print("\n=== Prompt Generation Test ===")
    prompt = build_user_prompt(
        dummy_episode["segments"][2],
        dummy_episode,
    )
    print(prompt)
    
    print("\n✓ GeminiConceptAnnotator test completed!")