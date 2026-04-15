"""
Stage 2: Gemini API Concept Annotation (via OpenRouter)
========================================================

CBM(Concept Bottleneck Model) 철학 — LaBo/CLG-CBM 스타일

=== Action Concepts ===
한 세그먼트에서 실행되는 동작을 순서대로 나열한 phrase 리스트.
각 phrase = 하나의 원자적 동작 step.
T5가 이 phrase들을 클러스터링해서 action concept pool 구축.

출력 예:
  ["arm extends forward toward target",
   "gripper opens wide preparing to grasp",
   "wrist descends and aligns above object"]

=== Scene Concepts ===
LaBo/CLG-CBM 스타일: 각 시각적 속성을 하나의 phrase로.
T5가 클러스터링해서 scene concept pool 구축.

출력 예:
  ["green colored object",
   "cylindrical shaped object",
   "object positioned at center of workspace"]
"""

import base64
import json
import time
import io
import os
from pathlib import Path
from typing import List, Optional, Dict


# ================================================================
# Action Prompt
# ================================================================

ACTION_SYSTEM = """You are a robotic manipulation expert. Describe a robot motion segment as an ORDERED LIST of atomic action steps.

OUTPUT RULES:
1. Return a JSON list "action_concepts" with 2-5 elements
2. Each element = ONE atomic motion phrase (NOT a full paragraph)
3. Order matters: list steps in execution order within this segment
4. Each phrase must describe ONE thing: which body part + what motion + direction/state
5. Use generic object words: "target object", "gripper", "arm", "wrist", "end effector"
   NEVER use specific object names (sponge, battery, cube...)
6. Each phrase should be 5-12 words, descriptive enough for T5 to distinguish

GOOD examples:
  "arm moves forward toward target object"
  "gripper opens wide in preparation"
  "wrist descends and aligns above target"
  "fingers close around the object firmly"
  "arm lifts object upward away from surface"
  "gripper releases and opens to drop object"
  "arm retracts to home position"

BAD examples:
  "The robot arm begins to move." ← too vague
  "grasp" ← single word, no context
  "The robot performs a complex manipulation involving..." ← too long

Respond ONLY with valid JSON. No markdown."""


def build_action_prompt(segment: dict, episode_info: dict) -> str:
    mv    = segment.get("mean_velocity", 0.0)
    vel   = "fast" if mv > 0.05 else ("slow/precise" if mv > 0.01 else "nearly stationary")
    total = len(episode_info.get("segments", []))
    robot = episode_info.get("robot_type", "unknown")
    task  = episode_info.get("task_description", "manipulation task")
    seg   = segment.get("segment_id", 0)
    is_bi = any(k in robot.lower() for k in ["aloha","gr1","bimanual","dual"])
    arm_note = "BIMANUAL — prefix steps with 'left arm' or 'right arm'" if is_bi else "SINGLE ARM"

    return f"""Describe this robot motion segment as ordered atomic action steps.

ROBOT: {robot} ({arm_note})
TASK GOAL: {task}
SEGMENT: step {seg+1} of {total}
  gripper={segment.get('gripper_state','?')} | direction={segment.get('dominant_direction','?')} | speed={vel}
  frames {segment.get('start_frame',0)}~{segment.get('end_frame',0)}

Output 2-5 ordered atomic steps for what happens IN THIS SEGMENT:
{{
  "action_concepts": [
    "first atomic step phrase",
    "second atomic step phrase",
    "third atomic step phrase"
  ]
}}"""


# ================================================================
# Scene Prompt
# ================================================================

SCENE_SYSTEM = """You are a robot vision expert. Describe visual scene attributes as a list of SHORT attribute phrases.

STYLE: LaBo/CLG-CBM concept descriptor style.
Each phrase = one visual attribute of the TARGET object.
Write like filling in "has ___" or "is ___" blanks.

OUTPUT RULES:
1. Return a JSON list "scene_concepts" with 2-4 elements
2. Each element describes ONE visual attribute: color, shape, OR location
3. Each phrase: 3-6 words only
4. Attribute categories to cover:
   - COLOR:    "[color] colored object"  e.g. "green colored object"
   - SHAPE:    "[shape] shaped object"   e.g. "cylindrical shaped object"
   - LOCATION: "object [position]"       e.g. "object at center of workspace"
                                         e.g. "object near the gripper"
                                         e.g. "object on left side"
5. If multiple colors visible, list each separately
6. DO NOT describe the robot arm or gripper actions

GOOD examples:
  "red colored object"
  "cube shaped object"
  "object positioned at center"
  "small rectangular object"
  "object held near gripper"
  "blue cylindrical container"

BAD examples:
  "robot approaches red cube at center" ← includes action
  "object" ← too vague
  "The target is a blue cube located..." ← too long

Respond ONLY with valid JSON. No markdown."""


def build_scene_prompt(segment: dict, episode_info: dict) -> str:
    task  = episode_info.get("task_description", "manipulation task")
    seg   = segment.get("segment_id", 0)
    total = len(episode_info.get("segments", []))

    return f"""Identify visual attribute phrases for the TARGET object in this frame.

TASK: {task}
SEGMENT: {seg+1} of {total} | gripper={segment.get('gripper_state','?')}

Describe color, shape, and location as separate short phrases:
{{
  "scene_concepts": [
    "color colored object",
    "shape shaped object",
    "object at location"
  ]
}}"""


# ================================================================
# Gemini via OpenRouter
# ================================================================

class GeminiConceptAnnotator:
    """
    세그먼트당 2회 독립 API 호출:
      Call 1 → action_concepts: ordered list of atomic motion phrases
      Call 2 → scene_concepts:  list of visual attribute phrases
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "google/gemini-2.0-flash-001",
        max_rpm: int = 30,
    ):
        self.api_key  = api_key or os.environ.get("OPENROUTER_API_KEY")
        self.model_name = model_name
        self._request_interval = 60.0 / max_rpm
        self._last_request_time = 0.0
        self._client = None

    def _init_client(self):
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
            print("  [WARNING] openai not installed: pip install openai")
            self._client = None

    def _rate_limit(self):
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < self._request_interval:
            time.sleep(self._request_interval - elapsed)
        self._last_request_time = time.time()

    def _encode_image(self, image_path: str) -> Optional[str]:
        try:
            import PIL.Image
            img = PIL.Image.open(image_path).resize((320, 240), PIL.Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="JPEG")
            return base64.b64encode(buf.getvalue()).decode("utf-8")
        except Exception:
            return None

    def _build_content(self, text: str, image_path: Optional[str]) -> list:
        content = [{"type": "text", "text": text}]
        if image_path:
            b64 = self._encode_image(image_path)
            if b64:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
                })
        return content

    def _call_api(self, system: str, user_content: list, max_tokens: int = 300) -> Optional[dict]:
        self._rate_limit()
        try:
            resp = self._client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user_content},
                ],
                response_format={"type": "json_object"},
                temperature=0.15,
                max_tokens=max_tokens,
            )
            parsed = json.loads(resp.choices[0].message.content)
            if isinstance(parsed, list):
                parsed = parsed[0] if parsed else {}
            return parsed if isinstance(parsed, dict) else None
        except Exception as e:
            print(f"    [ERROR] API call failed: {e}")
            return None

    def _extract_list(self, result: dict, key: str) -> List[str]:
        """dict에서 list 안전하게 추출 + 문자열 정규화"""
        val = result.get(key, [])
        if isinstance(val, str):
            # 문자열로 왔을 경우 분리 시도
            val = [v.strip() for v in val.split(".") if v.strip()]
        if not isinstance(val, list):
            val = []
        # 정규화: 너무 길거나 비어있는 원소 필터
        cleaned = []
        for item in val:
            s = str(item).strip()
            if 3 <= len(s) <= 120:  # 너무 짧거나 너무 긴 것 제외
                cleaned.append(s)
        return cleaned if cleaned else []

    def _call_action(self, image_path, segment, episode_info) -> dict:
        content = self._build_content(
            build_action_prompt(segment, episode_info), image_path
        )
        result = self._call_api(ACTION_SYSTEM, content, max_tokens=350)

        if result is None:
            return self._fallback_action(segment, episode_info)

        concepts = self._extract_list(result, "action_concepts")
        if not concepts:
            return self._fallback_action(segment, episode_info)

        return {
            "action_concepts": concepts,  # ordered list of atomic phrases
            # 하위 호환: 첫 번째 phrase를 단일 concept으로
            "action_concept": concepts[0],
            "action_description": " ".join(concepts),
        }

    def _call_scene(self, image_path, segment, episode_info) -> dict:
        content = self._build_content(
            build_scene_prompt(segment, episode_info), image_path
        )
        result = self._call_api(SCENE_SYSTEM, content, max_tokens=200)

        if result is None:
            return self._fallback_scene(segment, episode_info)

        concepts = self._extract_list(result, "scene_concepts")
        if not concepts:
            return self._fallback_scene(segment, episode_info)

        return {
            "scene_concepts": concepts,  # list of visual attribute phrases
            # 하위 호환
            "scene_concept": concepts[0],
            "scene_description": " | ".join(concepts),
        }

    def annotate_single(self, image_path, segment, episode_info) -> dict:
        self._init_client()
        if self._client is None:
            return {**self._fallback_action(segment, episode_info),
                    **self._fallback_scene(segment, episode_info)}
        action = self._call_action(image_path, segment, episode_info)
        scene  = self._call_scene(image_path, segment, episode_info)
        return {**action, **scene}

    # ── Fallback ─────────────────────────────────────────────────

    def _fallback_action(self, segment, episode_info) -> dict:
        gripper   = segment.get("gripper_state", "open")
        direction = segment.get("dominant_direction", "forward")
        seg_id    = segment.get("segment_id", 0)
        total     = len(episode_info.get("segments", []))

        if seg_id == 0:
            concepts = [
                "arm extends forward toward target object",
                "gripper opens wide in preparation",
            ]
        elif gripper == "transitioning" and seg_id < total // 2:
            concepts = [
                "arm descends and aligns above target object",
                "fingers close around object to secure grip",
            ]
        elif gripper == "closed" and direction in ("up", "raise"):
            concepts = [
                "arm lifts object upward away from surface",
                "gripper maintains firm hold during lift",
            ]
        elif gripper == "closed":
            concepts = [
                f"arm carries object in {direction} direction",
                "gripper holds object securely during transport",
            ]
        elif gripper == "transitioning" and seg_id >= total // 2:
            concepts = [
                "arm positions object at target location",
                "gripper opens to release object",
            ]
        elif seg_id == total - 1:
            concepts = [
                "arm retracts away from target location",
                "gripper returns to open resting state",
            ]
        else:
            concepts = [
                f"arm repositions in {direction} direction",
                "gripper maintains current state",
            ]

        return {
            "action_concepts":    concepts,
            "action_concept":     concepts[0],
            "action_description": " ".join(concepts),
        }

    def _fallback_scene(self, segment, episode_info) -> dict:
        concepts = [
            "object positioned on workspace surface",
            "target object at center of field of view",
        ]
        return {
            "scene_concepts":    concepts,
            "scene_concept":     concepts[0],
            "scene_description": " | ".join(concepts),
        }

    # ── 이미지 추출 ───────────────────────────────────────────────

    def _extract_frame_image(self, dataset_root, episode_id, frame_idx,
                             camera="observation.images.top"):
        root = Path(dataset_root) if dataset_root else None
        if root is None:
            return None
        camera_dir = camera.replace(".", "/")
        fp = root / "data" / camera_dir / f"frame_{frame_idx:06d}.png"
        if fp.exists():
            return str(fp)
        vp = root / "videos" / f"{camera}_episode_{episode_id:06d}.mp4"
        if vp.exists():
            return self._extract_frame_from_video(str(vp), frame_idx)
        return None

    def _extract_frame_from_video(self, video_path, frame_idx):
        try:
            import cv2
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            cap.release()
            if ret:
                tmp = f"/tmp/cbmvla_{frame_idx}.jpg"
                cv2.imwrite(tmp, frame)
                return tmp
        except ImportError:
            pass
        return None

    # ── 전체 annotation ───────────────────────────────────────────

    def annotate_segments(self, segments, dataset_root=None,
                          output_path=None, batch_size=10) -> List[dict]:
        all_annotations = []
        processed_keys  = set()

        if output_path and Path(output_path).exists():
            with open(output_path) as f:
                all_annotations = json.load(f)
            processed_keys = {(a["episode_id"], a["segment_id"])
                              for a in all_annotations}
            print(f"  Loaded {len(all_annotations)} existing annotations")

        total     = sum(len(ep["segments"]) for ep in segments)
        processed = len(processed_keys)
        print(f"  Total segments: {total} (≈{total*2} API calls)")
        print(f"  Remaining: {total - processed}")

        for ep in segments:
            episode_id = ep["episode_id"]
            for seg in ep["segments"]:
                seg_id = seg["segment_id"]
                if (episode_id, seg_id) in processed_keys:
                    continue

                image_path = None
                if dataset_root:
                    image_path = self._extract_frame_image(
                        dataset_root, episode_id, seg["representative_frame"])

                annotation = self.annotate_single(image_path, seg, ep)
                annotation.update({
                    "episode_id":           episode_id,
                    "segment_id":           seg_id,
                    "start_frame":          seg["start_frame"],
                    "end_frame":            seg["end_frame"],
                    "representative_frame": seg["representative_frame"],
                    "task_description":     ep.get("task_description",""),
                    "robot_type":           ep.get("robot_type","unknown"),
                    "has_image":            image_path is not None,
                })
                all_annotations.append(annotation)
                processed += 1

                if processed % batch_size == 0:
                    print(f"    Processed {processed}/{total} segments")
                    if output_path:
                        with open(output_path, "w") as f:
                            json.dump(all_annotations, f, indent=2,
                                      ensure_ascii=False)

        if output_path:
            with open(output_path, "w") as f:
                json.dump(all_annotations, f, indent=2, ensure_ascii=False)

        print(f"  Annotation complete: {len(all_annotations)} segments")
        return all_annotations

    def estimate_cost(self, num_segments, model="gemini-2.0-flash-001") -> dict:
        calls       = num_segments * 2
        input_toks  = calls * 650
        output_toks = calls * 180
        pricing = {
            "gemini-2.0-flash-001":      {"input": 0.10, "output": 0.40},
            "gemini-2.0-flash-lite-001": {"input": 0.075,"output": 0.30},
            "gemini-2.5-flash":          {"input": 0.30, "output": 2.50},
        }
        p  = pricing.get(model, {"input": 0.10, "output": 0.40})
        ic = (input_toks  / 1_000_000) * p["input"]
        oc = (output_toks / 1_000_000) * p["output"]
        return {
            "model": model, "num_segments": num_segments, "api_calls": calls,
            "total_input_tokens":   input_toks,
            "total_output_tokens":  output_toks,
            "input_cost_usd":       round(ic, 2),
            "output_cost_usd":      round(oc, 2),
            "total_cost_usd":       round(ic + oc, 2),
            "batch_discount_50pct": round((ic + oc) * 0.5, 2),
        }


if __name__ == "__main__":
    a = GeminiConceptAnnotator()

    print("=== Cost (17,400 episodes × 4.5 segs × 2 calls) ===")
    c = a.estimate_cost(78300, "gemini-2.0-flash-001")
    print(f"  ${c['total_cost_usd']} ({c['api_calls']:,} calls)")

    print("\n=== Fallback Test ===")
    ep = {
        "episode_id": 0, "robot_type": "so101",
        "task_description": "pick the battery and insert into slot",
        "total_frames": 400,
        "segments": [
            {"segment_id":0,"start_frame":0,"end_frame":100,
             "gripper_state":"open","dominant_direction":"forward",
             "mean_velocity":0.04,"trigger":"start","representative_frame":50},
            {"segment_id":1,"start_frame":100,"end_frame":160,
             "gripper_state":"transitioning","dominant_direction":"down",
             "mean_velocity":0.008,"trigger":"gripper","representative_frame":130},
            {"segment_id":2,"start_frame":160,"end_frame":280,
             "gripper_state":"closed","dominant_direction":"up",
             "mean_velocity":0.05,"trigger":"velocity","representative_frame":220},
        ],
    }
    for seg in ep["segments"]:
        ra = a._fallback_action(seg, ep)
        rs = a._fallback_scene(seg, ep)
        print(f"\n  seg{seg['segment_id']}:")
        print(f"    action_concepts: {ra['action_concepts']}")
        print(f"    scene_concepts:  {rs['scene_concepts']}")