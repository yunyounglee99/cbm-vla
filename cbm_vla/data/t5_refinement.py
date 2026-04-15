"""
Stage 3: T5 Concept Refinement — LaBo/CLG-CBM Style
=====================================================

Gemini가 생성한 phrase 리스트에서 concept pool을 구축합니다.

=== Action Concept Pool ===
입력: ordered list of atomic motion phrases
  seg0: ["arm extends forward toward target",
         "gripper opens wide in preparation"]
  seg1: ["arm descends and aligns above target",
         "fingers close around object to secure grip"]
  ...

T5가 의미적으로 유사한 phrase를 클러스터링:
  "arm extends forward toward target"    ─┐
  "arm moves forward approaching object" ─┘→ approach_target
  "gripper opens wide"                   ─┐
  "gripper opens in preparation"         ─┘→ gripper_open_prepare
  "fingers close around object"          ─┐
  "gripper closes to grasp"              ─┘→ grasp_object

=== Scene Concept Pool ===
입력: list of visual attribute phrases
  ["green colored object",
   "cylindrical shaped object",
   "object at center of workspace"]

T5가 클러스터링:
  "green colored object"   ─┐
  "green object"           ─┘→ green_colored
  "cylindrical shaped"     ─┐
  "cylinder-like object"   ─┘→ cylindrical_shape
  "object at center"       ─┐
  "centered in frame"      ─┘→ center_position
"""

import json
import numpy as np
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple


class ConceptRefiner:
    """
    LaBo/CLG-CBM 스타일 concept pool 구축.

    Action: T5(phrase) 클러스터링 — ordered motion phrases
    Scene:  T5(phrase) 클러스터링 — visual attribute phrases
    """

    def __init__(
        self,
        model_name: str = "google/flan-t5-base",
        similarity_threshold: float = 0.85,
        max_concepts: int = 50,
        max_action_concepts: int = 50,
        max_scene_concepts: int = 50,
        min_frequency: int = 2,
    ):
        self.model_name           = model_name
        self.similarity_threshold = similarity_threshold
        self.max_action_concepts  = max_action_concepts or max_concepts
        self.max_scene_concepts   = max_scene_concepts  or max_concepts
        self.min_frequency        = min_frequency
        self._encoder   = None
        self._tokenizer = None

    # ── T5 / TF-IDF 인코더 ────────────────────────────────────────

    def _init_encoder(self):
        if self._encoder is not None:
            return
        try:
            from transformers import T5EncoderModel, T5Tokenizer
            import torch
            print(f"  Loading T5 encoder: {self.model_name}")
            self._tokenizer = T5Tokenizer.from_pretrained(self.model_name)
            self._encoder   = T5EncoderModel.from_pretrained(self.model_name)
            self._encoder.eval()
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._encoder = self._encoder.to(self._device)
            print(f"  T5 encoder loaded on {self._device}")
        except ImportError:
            print("  [WARNING] transformers not available. Using TF-IDF.")
            self._encoder = None

    def _encode_texts(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        self._init_encoder()
        if self._encoder is None:
            return self._tfidf_encode(texts)
        import torch
        all_embs = []
        for i in range(0, len(texts), batch_size):
            batch  = texts[i:i + batch_size]
            inputs = self._tokenizer(batch, return_tensors="pt", padding=True,
                                     truncation=True, max_length=64).to(self._device)
            with torch.no_grad():
                out  = self._encoder(**inputs)
                mask = inputs["attention_mask"].unsqueeze(-1).float()
                emb  = (out.last_hidden_state * mask).sum(1) / mask.sum(1)
                emb  = emb / (emb.norm(dim=-1, keepdim=True) + 1e-8)
                all_embs.append(emb.cpu().numpy())
        return np.concatenate(all_embs, axis=0)

    def _tfidf_encode(self, texts: List[str],
                      vocab: Optional[dict] = None) -> np.ndarray:
        if vocab is None:
            all_words = set()
            for t in texts:
                all_words.update(t.lower().split())
            vocab = {w: i for i, w in enumerate(sorted(all_words))}
        dim  = max(len(vocab), 1)
        embs = np.zeros((len(texts), dim))
        for i, t in enumerate(texts):
            for w in t.lower().split():
                if w in vocab:
                    embs[i, vocab[w]] += 1
        norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8
        return embs / norms

    def _encode_scene_phrases(self, phrases: List[str],
                               vocab: Optional[dict] = None) -> np.ndarray:
        """
        Scene phrase 전용 인코딩 — 항상 TF-IDF 사용.
        vocab을 외부에서 넘기면 동일 vocabulary로 인코딩 (차원 일치 보장).

        이유: T5는 "blue colored object" ≈ "black colored object"로 봄.
             TF-IDF는 색상 단어가 다르면 유사도 0.33 → 완전 분리.
        """
        return self._tfidf_encode(phrases, vocab)  # 항상 TF-IDF

    # ── 클러스터링 ────────────────────────────────────────────────

    # 반의어 쌍 (PAIR: side_a vs side_b)
    # 같은 클러스터에 side_a와 side_b 단어가 모두 있으면 conflict
    _ANTONYM_PAIRS = [
        (
            {"up", "upward", "ascend", "ascending", "ascends", "rise", "rising",
             "above", "lift", "lifts", "raise", "raises"},
            {"down", "downward", "descend", "descends", "descending", "descent",
             "lower", "lowers", "drop", "drops"},
        ),
        (
            {"clockwise"},
            {"counterclockwise", "counter"},
        ),
        (
            {"left", "leftward"},
            {"right", "rightward"},
        ),
        (
            {"forward", "extend", "extends"},
            {"backward", "away", "retract", "retracts"},
        ),
    ]

    def _get_phrase_direction_label(self, phrase: str) -> Optional[str]:
        """phrase에서 방향 레이블 반환. 반의어 쌍에서 어느 쪽인지 식별."""
        words = set(phrase.lower().replace("-","").split())
        for i, (side_a, side_b) in enumerate(self._ANTONYM_PAIRS):
            if words & side_a:
                return f"pair{i}_a"
            if words & side_b:
                return f"pair{i}_b"
        return None

    # 다른 축(axis) motion 단어 그룹
    # 수직, 수평, 회전 중 2개 이상의 축이 혼재하면 분리
    _MOTION_AXES = [
        {"up","upward","ascend","ascends","rise","above","lift","raise",
         "down","downward","descend","descends","descent","lower","drop"},  # 수직
        {"forward","extend","extends","approach","approaches",
         "backward","retract","retracts","away"},                           # 전후
        {"left","leftward","right","rightward","lateral"},                  # 좌우
        {"clockwise","counterclockwise","rotate","rotates","rotating"},     # 회전
    ]

    def _get_axes(self, phrase: str) -> set:
        words = set(phrase.lower().replace("-","").split())
        axes  = set()
        for i, axis_words in enumerate(self._MOTION_AXES):
            if words & axis_words:
                axes.add(i)
        return axes

    def _has_direction_conflict(self, phrases: List[str]) -> bool:
        """반의어 또는 다른 축의 motion이 섞여있는지 확인"""
        # 1. 반의어 쌍 체크
        for side_a, side_b in self._ANTONYM_PAIRS:
            has_a = any(set(p.lower().replace("-","").split()) & side_a for p in phrases)
            has_b = any(set(p.lower().replace("-","").split()) & side_b for p in phrases)
            if has_a and has_b:
                return True
        # 2. 다른 축 motion 혼재 체크 (수직+수평, 수직+회전 등)
        all_axes: set = set()
        for p in phrases:
            all_axes |= self._get_axes(p)
        if len(all_axes) >= 2:
            return True
        return False

    def _split_direction_conflicts(self, clusters: List[dict]) -> List[dict]:
        """
        반의어 또는 다른 축의 motion이 섞인 클러스터를 분리.
        """
        result = []
        for cl in clusters:
            if len(cl["members"]) <= 1 or not self._has_direction_conflict(cl["members"]):
                result.append(cl)
                continue

            # 각 phrase의 axis set 기반으로 그룹화
            # axis가 같은 것끼리 묶고, 다른 axis는 다른 그룹
            axis_groups: Dict[str, List[str]] = {}

            for phrase in cl["members"]:
                axes  = self._get_axes(phrase)
                # 반의어 쌍 레이블도 고려
                label = self._get_phrase_direction_label(phrase)

                # axis가 없으면 label로, 둘 다 없으면 "other"
                if axes:
                    key = "_".join(str(a) for a in sorted(axes))
                elif label:
                    key = label
                else:
                    key = "other"

                axis_groups.setdefault(key, []).append(phrase)

            for phrases_in_group in axis_groups.values():
                result.append({
                    "representative": phrases_in_group[0],
                    "members":        phrases_in_group,
                    "size":           len(phrases_in_group),
                })

        return result

    def _cluster(
        self,
        phrases: List[str],
        embeddings: np.ndarray,
        threshold: float,
    ) -> List[dict]:
        """Agglomerative clustering (Union-Find)"""
        N = len(phrases)
        if N == 0:
            return []

        sim    = embeddings @ embeddings.T
        parent = list(range(N))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        for i in range(N):
            for j in range(i + 1, N):
                if sim[i, j] > threshold:
                    union(i, j)

        groups = defaultdict(list)
        for i in range(N):
            groups[find(i)].append(i)

        clusters = []
        for _, members in sorted(groups.items()):
            member_embs = embeddings[members]
            centroid    = member_embs.mean(0)
            centroid   /= (np.linalg.norm(centroid) + 1e-8)
            best        = members[int(np.argmax(member_embs @ centroid))]
            clusters.append({
                "representative": phrases[best],
                "members":        [phrases[m] for m in members],
                "size":           len(members),
            })
        clusters.sort(key=lambda x: -x["size"])
        return clusters

    def _build_pool_from_clusters(
        self,
        clusters: List[dict],
        phrase_counts: Counter,
        concept_type: str,
        max_size: int,
    ) -> List[dict]:
        """클러스터 → concept pool"""
        total = sum(phrase_counts.values())
        pool  = []

        for cl in clusters:
            # 클러스터 내 전체 빈도 합산
            freq = sum(phrase_counts.get(p, 0) for p in cl["members"])
            if freq < self.min_frequency:
                continue
            if len(pool) >= max_size:
                break

            # 대표 이름: representative phrase → snake_case 단축
            rep   = cl["representative"]
            base  = "_".join(rep.lower().split()[:4])  # 앞 4단어
            # 중복 이름 방지: 이미 사용된 이름이면 단어 추가
            name  = base
            used_names = {c["name"] for c in pool}
            extra = 5
            while name in used_names and extra <= len(rep.lower().split()):
                name = "_".join(rep.lower().split()[:extra])
                extra += 1
            if name in used_names:
                name = f"{base}_{len(pool)}"

            # 상위 5개 변형
            variants = sorted(
                cl["members"],
                key=lambda p: -phrase_counts.get(p, 0)
            )[:5]

            pool.append({
                "concept_id":    len(pool),
                "concept_type":  concept_type,
                "name":          name,
                "description":   rep,
                "frequency":     freq,
                "frequency_pct": round(100 * freq / max(total, 1), 1),
                "cluster_size":  cl["size"],
                "variants":      variants,
            })

        return pool

    # ── 메인 파이프라인 ───────────────────────────────────────────

    def refine_and_build(
        self,
        raw_concepts: List[dict],
        segments: List[dict],
    ) -> Tuple[List[dict], List[dict], List[dict]]:
        """
        LaBo/CLG-CBM 스타일 Dual Concept Pool 구축.

        Returns:
            action_pool:   ordered motion phrase clusters
            scene_pool:    visual attribute phrase clusters
            final_dataset: segment별 concept ID 매핑
        """
        if not raw_concepts:
            print("  [WARNING] No raw concepts.")
            return [], [], []

        # ── 1. phrase 수집 ────────────────────────────────────────
        # Action: 각 세그먼트의 ordered list 원소들을 모두 수집
        action_phrase_counts: Counter = Counter()
        # Scene: 각 세그먼트의 attribute phrase들을 모두 수집
        scene_phrase_counts:  Counter = Counter()

        for ann in raw_concepts:
            # action_concepts는 ordered list
            a_phrases = ann.get("action_concepts", [])
            if isinstance(a_phrases, str):
                a_phrases = [a_phrases]
            for p in a_phrases:
                if isinstance(p, str) and p.strip():
                    action_phrase_counts[p.strip().lower()] += 1

            # scene_concepts는 attribute phrase list
            s_phrases = ann.get("scene_concepts", [])
            if isinstance(s_phrases, str):
                s_phrases = [s_phrases]
            for p in s_phrases:
                if isinstance(p, str) and p.strip():
                    scene_phrase_counts[p.strip().lower()] += 1

        print(f"  Unique action phrases: {len(action_phrase_counts)}")
        print(f"  Unique scene  phrases: {len(scene_phrase_counts)}")

        # ── 2. T5 임베딩 ─────────────────────────────────────────
        print("  Computing T5 embeddings...")

        a_phrases = list(action_phrase_counts.keys())
        s_phrases = list(scene_phrase_counts.keys())

        # Action: T5 — 같은 의미의 다른 표현 감지 (paraphrase 합침)
        a_embs = self._encode_texts(a_phrases) if a_phrases else np.zeros((0,1))

        # Scene: TF-IDF — 전체 scene phrase로 공유 vocabulary 먼저 구축
        # → s_embs와 s_pool_embs가 같은 vocab으로 인코딩되어 차원 일치 보장
        if s_phrases:
            all_words = set()
            for p in s_phrases:
                all_words.update(p.lower().split())
            self._scene_shared_vocab = {w: i for i, w in enumerate(sorted(all_words))}
            s_embs = self._encode_scene_phrases(s_phrases, self._scene_shared_vocab)
        else:
            self._scene_shared_vocab = {}
            s_embs = np.zeros((0, 1))

        print(f"  Action emb (T5):    {a_embs.shape}")
        print(f"  Scene  emb (TF-IDF):{s_embs.shape} vocab={len(self._scene_shared_vocab)}")

        # ── 3. 클러스터링 ─────────────────────────────────────────
        # Action: 0.95 — 거의 동일 표현만 합침 (방향 혼합 방지)
        # Scene:  0.88 — 같은 속성 다르게 표현한 것만 합침
        a_threshold = min(self.similarity_threshold + 0.10, 0.97)
        s_threshold = min(self.similarity_threshold + 0.03, 0.92)
        print(f"  Clustering (action={a_threshold:.2f}, scene={s_threshold:.2f})...")

        a_clusters_raw = self._cluster(a_phrases, a_embs, a_threshold) if a_phrases else []
        # 방향 반의어가 섞인 클러스터 강제 분리
        a_clusters = self._split_direction_conflicts(a_clusters_raw)
        s_clusters = self._cluster(s_phrases, s_embs, s_threshold) if s_phrases else []
        print(f"  Action clusters after direction split: {len(a_clusters)}")
        print(f"  Action clusters: {len(a_clusters)}")
        print(f"  Scene  clusters: {len(s_clusters)}")

        # ── 4. Pool 구축 ──────────────────────────────────────────
        action_pool = self._build_pool_from_clusters(
            a_clusters, action_phrase_counts, "action", self.max_action_concepts
        )
        scene_pool = self._build_pool_from_clusters(
            s_clusters, scene_phrase_counts, "scene", self.max_scene_concepts
        )
        print(f"  Action pool: {len(action_pool)}")
        print(f"  Scene  pool: {len(scene_pool)}")

        # ── 5. Pool 임베딩 (매핑용) ───────────────────────────────
        # action pool: T5
        a_pool_embs = (self._encode_texts([c["description"] for c in action_pool])
                       if action_pool else np.zeros((0,1)))
        # scene pool: 같은 shared_vocab으로 TF-IDF (차원 일치)
        scene_vocab = getattr(self, "_scene_shared_vocab", {})
        s_pool_embs = (self._encode_scene_phrases([c["description"] for c in scene_pool], scene_vocab)
                       if scene_pool and scene_vocab
                       else np.zeros((len(scene_pool), max(len(scene_vocab), 1))))

        # ── 6. phrase → pool_id 매핑 ──────────────────────────────
        def _build_map(phrases, pool, pool_embs, phrase_embs) -> Dict[str, int]:
            """각 raw phrase를 가장 가까운 pool concept에 매핑"""
            name2id: Dict[str, int] = {}
            # 직접 매핑
            for c in pool:
                name2id[c["description"]] = c["concept_id"]
                for v in c["variants"]:
                    name2id[v] = c["concept_id"]
            # 임베딩 유사도 매핑
            if len(pool_embs) > 0 and len(phrase_embs) > 0:
                p2idx = {p: i for i, p in enumerate(phrases)}
                for p in phrases:
                    if p not in name2id and p in p2idx:
                        sims = pool_embs @ phrase_embs[p2idx[p]]
                        name2id[p] = int(np.argmax(sims))
            return name2id

        a_map = _build_map(a_phrases, action_pool, a_pool_embs, a_embs)
        # scene 매핑도 TF-IDF 임베딩 사용 (s_embs가 이미 TF-IDF)
        s_map = _build_map(s_phrases, scene_pool,  s_pool_embs, s_embs)

        # ── 7. 최종 dataset 구성 ──────────────────────────────────
        print("  Building final dataset...")
        dataset = []

        for ann in raw_concepts:
            # action: ordered list → 각 phrase의 pool_id (순서 유지)
            a_phrases_raw = ann.get("action_concepts", [])
            if isinstance(a_phrases_raw, str):
                a_phrases_raw = [a_phrases_raw]
            active_action_ids = []
            for p in a_phrases_raw:
                p = p.strip().lower()
                if p in a_map:
                    cid = a_map[p]
                    if cid not in active_action_ids:
                        active_action_ids.append(cid)

            # scene: attribute phrases → 각각의 pool_id
            s_phrases_raw = ann.get("scene_concepts", [])
            if isinstance(s_phrases_raw, str):
                s_phrases_raw = [s_phrases_raw]
            active_scene_ids = []
            for p in s_phrases_raw:
                p = p.strip().lower()
                if p in s_map:
                    cid = s_map[p]
                    if cid not in active_scene_ids:
                        active_scene_ids.append(cid)

            dataset.append({
                "episode_id":           ann["episode_id"],
                "segment_id":           ann["segment_id"],
                "start_frame":          ann["start_frame"],
                "end_frame":            ann["end_frame"],
                "representative_frame": ann["representative_frame"],
                "task_description":     ann.get("task_description", ""),
                "robot_type":           ann.get("robot_type", "unknown"),
                # action
                "action_concepts":      a_phrases_raw,       # 원본 ordered phrases
                "active_action_ids":    active_action_ids,    # ordered pool IDs
                "action_description":   ann.get("action_description",""),
                # scene
                "scene_concepts":       s_phrases_raw,        # 원본 attribute phrases
                "active_scene_ids":     active_scene_ids,     # pool IDs
                "scene_description":    ann.get("scene_description",""),
                # 하위 호환
                "action_concept_id":    active_action_ids[0] if active_action_ids else 0,
                "scene_concept_id":     active_scene_ids[0]  if active_scene_ids  else 0,
            })

        # 에피소드 순서 정보
        ep_groups = defaultdict(list)
        for entry in dataset:
            ep_groups[entry["episode_id"]].append(entry)
        for ep_id, entries in ep_groups.items():
            entries.sort(key=lambda x: x["segment_id"])
            for order, entry in enumerate(entries):
                entry["order_in_episode"]           = order
                entry["total_segs_in_episode"]      = len(entries)
                entry["episode_action_concept_ids"] = [
                    aid for e in entries for aid in e["active_action_ids"]]
                entry["episode_scene_concept_ids"]  = [
                    sid for e in entries for sid in e["active_scene_ids"]]

        return action_pool, scene_pool, dataset


if __name__ == "__main__":
    print("Testing LaBo-style ConceptRefiner...")
    refiner = ConceptRefiner(similarity_threshold=0.85, min_frequency=1)

    dummy = []
    for ep_id in range(5):
        segs = [
            {"action_concepts": ["arm extends forward toward target object",
                                  "gripper opens wide in preparation"],
             "scene_concepts":  ["green colored object",
                                  "cylindrical shaped object",
                                  "object at center of workspace"]},
            {"action_concepts": ["arm descends and aligns above target",
                                  "fingers close around object to secure grip",
                                  "gripper tightens for stable hold"],
             "scene_concepts":  ["green colored object",
                                  "cylindrical shaped object",
                                  "object near the gripper"]},
            {"action_concepts": ["arm lifts object upward away from surface",
                                  "gripper maintains firm hold during lift"],
             "scene_concepts":  ["green colored object",
                                  "object held above surface"]},
        ]
        for seg_id, s in enumerate(segs):
            dummy.append({
                "episode_id": ep_id, "segment_id": seg_id,
                "start_frame": seg_id*100, "end_frame": (seg_id+1)*100,
                "representative_frame": seg_id*100+50,
                "action_concepts":    s["action_concepts"],
                "action_description": " ".join(s["action_concepts"]),
                "scene_concepts":     s["scene_concepts"],
                "scene_description":  " | ".join(s["scene_concepts"]),
                "task_description":   "pick and place green cylinder",
                "robot_type":         "so101",
            })

    action_pool, scene_pool, dataset = refiner.refine_and_build(dummy, [])

    print(f"\n=== Action Pool ({len(action_pool)}) ===")
    for c in action_pool:
        print(f"  [{c['concept_id']:2d}] {c['name']:35s} freq={c['frequency']:3d} | {c['description'][:60]}")

    print(f"\n=== Scene Pool ({len(scene_pool)}) ===")
    for c in scene_pool:
        print(f"  [{c['concept_id']:2d}] {c['name']:30s} freq={c['frequency']:3d} | {c['description']}")

    print(f"\n=== Dataset Sample (ep0, seg0) ===")
    e = dataset[0]
    print(f"  action_concepts:   {e['action_concepts']}")
    print(f"  active_action_ids: {e['active_action_ids']}")
    print(f"  scene_concepts:    {e['scene_concepts']}")
    print(f"  active_scene_ids:  {e['active_scene_ids']}")
    print("\n✓ Done")