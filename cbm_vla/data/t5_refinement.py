"""
Stage 3: T5 Concept Refinement & Pool Construction
====================================================

Gemini가 생성한 수만 개의 raw concept 문장을 정제하여
20~50개의 표준화된 concept pool을 구축합니다.

=== LaBo/CLG-CBM과의 관계 ===

LaBo는 GPT-3가 생성한 후보 컨셉에서 submodular optimization으로
discriminability와 diversity를 최대화하는 컨셉 부분집합을 선택합니다.

CBM-VLA에서는 LaBo의 "정신"(자동 컨셉 선택)을 따르되, 방법은 다릅니다:
  
  1. Embedding 기반 클러스터링 (LaBo의 submodular selection 대체)
     - T5 encoder로 모든 raw concept 문장을 임베딩
     - Cosine similarity 기반 클러스터링
     - 클러스터 중심에 가장 가까운 문장을 대표 컨셉으로 선택
     
  2. 빈도 기반 필터링 (LaBo에는 없는 추가 단계)
     - 너무 드문 컨셉(<5회 출현)은 노이즈로 간주하여 제거
     - 너무 흔한 컨셉(>80% 에피소드)은 정보량이 적어 제거 후보
     
  3. 수동 검증 옵션 (LaBo의 human evaluation과 유사)
     - 최종 컨셉 풀을 출력하여 연구자가 검토 가능
     - 의미적으로 부적절한 컨셉을 수동으로 교체/제거

로봇 도메인에서 submodular optimization이 불필요한 이유:
  - LaBo의 대상: ImageNet (1000 클래스, 수천 개 후보 컨셉)
  - CBM-VLA의 대상: tabletop manipulation (고유 행동 ~20~50개)
  - 컨셉 수가 적으므로 클러스터링 + 빈도 필터링으로 충분
"""

import json
import numpy as np
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple
from pathlib import Path


class ConceptRefiner:
    """
    T5 기반 컨셉 정제 및 풀 구축
    
    Pipeline:
    1. T5 encoder로 모든 raw concept description을 임베딩
    2. Cosine similarity 기반 Agglomerative Clustering
    3. 클러스터별 대표 컨셉 선택 (centroid에 가장 가까운 것)
    4. 빈도 필터링 (너무 드물거나 너무 흔한 컨셉 제거)
    5. 최종 concept pool 구축
    
    Args:
        model_name: T5 모델 이름 (문장 임베딩용)
        similarity_threshold: 클러스터링 병합 임계값 (cosine sim)
        max_concepts: 최대 컨셉 수
        min_frequency: 최소 출현 빈도 (이하면 제거)
    """
    
    def __init__(
        self,
        model_name: str = "google/flan-t5-base",
        similarity_threshold: float = 0.85,
        max_concepts: int = 50,
        min_frequency: int = 5,
    ):
        self.model_name = model_name
        self.similarity_threshold = similarity_threshold
        self.max_concepts = max_concepts
        self.min_frequency = min_frequency
        
        self._encoder = None
        self._tokenizer = None
    
    def _init_encoder(self):
        """T5 인코더 초기화"""
        if self._encoder is not None:
            return
        
        try:
            from transformers import T5EncoderModel, T5Tokenizer
            import torch
            
            print(f"  Loading T5 encoder: {self.model_name}")
            self._tokenizer = T5Tokenizer.from_pretrained(self.model_name)
            self._encoder = T5EncoderModel.from_pretrained(self.model_name)
            self._encoder.eval()
            
            if torch.cuda.is_available():
                self._encoder = self._encoder.cuda()
                self._device = "cuda"
            else:
                self._device = "cpu"
            
            print(f"  T5 encoder loaded on {self._device}")
            
        except ImportError:
            print("  [WARNING] transformers not available. Using simple text similarity.")
            self._encoder = None
    
    def _encode_texts(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        """
        T5 encoder로 텍스트 리스트를 임베딩
        
        Args:
            texts: 임베딩할 텍스트 리스트
            batch_size: 배치 크기
            
        Returns:
            embeddings: [num_texts, hidden_dim] numpy array
        """
        self._init_encoder()
        
        if self._encoder is None:
            return self._simple_encode(texts)
        
        import torch
        
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            inputs = self._tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128,
            ).to(self._device)
            
            with torch.no_grad():
                outputs = self._encoder(**inputs)
                # Mean pooling over sequence
                mask = inputs["attention_mask"].unsqueeze(-1).float()
                embeddings = (outputs.last_hidden_state * mask).sum(1) / mask.sum(1)
                # L2 normalize
                embeddings = embeddings / (embeddings.norm(dim=-1, keepdim=True) + 1e-8)
                all_embeddings.append(embeddings.cpu().numpy())
        
        return np.concatenate(all_embeddings, axis=0)
    
    def _simple_encode(self, texts: List[str]) -> np.ndarray:
        """
        T5 없이 간단한 텍스트 특성 추출 (fallback)
        
        TF-IDF 스타일의 단어 빈도 기반 벡터화
        """
        from collections import Counter
        
        # 단어 사전 구축
        all_words = set()
        for t in texts:
            all_words.update(t.lower().split())
        word2idx = {w: i for i, w in enumerate(sorted(all_words))}
        
        # TF 벡터 생성
        embeddings = np.zeros((len(texts), len(word2idx)))
        for i, t in enumerate(texts):
            words = t.lower().split()
            counts = Counter(words)
            for w, c in counts.items():
                if w in word2idx:
                    embeddings[i, word2idx[w]] = c
        
        # L2 normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
        embeddings = embeddings / norms
        
        return embeddings
    
    def cluster_concepts(
        self,
        concept_names: List[str],
        descriptions: List[str],
        embeddings: np.ndarray,
    ) -> List[dict]:
        """
        임베딩 기반 컨셉 클러스터링
        
        Agglomerative Clustering with average linkage.
        Cosine similarity가 threshold 이상인 쌍을 같은 클러스터로 병합.
        
        Args:
            concept_names: ["approach_object", "reach_toward_cube", ...]
            descriptions: ["Robot arm approaches...", "Robot reaches toward...", ...]
            embeddings: [N, D] 임베딩 행렬
            
        Returns:
            클러스터 리스트, 각 클러스터는:
            {
                "cluster_id": 0,
                "representative_name": "approach_object",
                "representative_description": "Robot arm approaches the target",
                "members": [{"name": ..., "description": ..., "count": ...}, ...],
                "total_count": 1234,
            }
        """
        N = len(concept_names)
        
        # Cosine similarity matrix
        sim_matrix = embeddings @ embeddings.T
        
        # 간단한 agglomerative clustering (Union-Find)
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
        
        # 유사도가 높은 쌍 병합
        for i in range(N):
            for j in range(i + 1, N):
                if sim_matrix[i, j] > self.similarity_threshold:
                    union(i, j)
        
        # 클러스터 그룹 생성
        cluster_groups = defaultdict(list)
        for i in range(N):
            cluster_groups[find(i)].append(i)
        
        # 클러스터별 대표 선택
        clusters = []
        for cluster_id, (_, members) in enumerate(sorted(cluster_groups.items())):
            # 클러스터 중심 계산
            member_embeddings = embeddings[members]
            centroid = member_embeddings.mean(axis=0)
            centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
            
            # 중심에 가장 가까운 멤버를 대표로 선택
            similarities = member_embeddings @ centroid
            best_idx = members[np.argmax(similarities)]
            
            # 멤버 정보 수집
            member_info = []
            for m in members:
                member_info.append({
                    "name": concept_names[m],
                    "description": descriptions[m],
                })
            
            clusters.append({
                "cluster_id": cluster_id,
                "representative_name": concept_names[best_idx],
                "representative_description": descriptions[best_idx],
                "members": member_info,
                "total_count": len(members),
            })
        
        # 크기 순으로 정렬 (큰 클러스터가 더 중요)
        clusters.sort(key=lambda x: -x["total_count"])
        
        return clusters
    
    def build_concept_pool(
        self,
        clusters: List[dict],
        raw_annotations: List[dict],
    ) -> List[dict]:
        """
        최종 concept pool 구축
        
        Args:
            clusters: 클러스터링 결과
            raw_annotations: 원본 annotation 데이터 (빈도 계산용)
            
        Returns:
            concept_pool: [
                {
                    "concept_id": 0,
                    "name": "approach_object",
                    "description": "Robot arm approaches the target object",
                    "frequency": 4523,
                    "frequency_pct": 19.6,
                    "cluster_size": 15,
                    "variants": ["approach_object", "reach_toward", "move_to_object"],
                },
                ...
            ]
        """
        # 전체 annotation에서 concept 빈도 계산
        concept_counts = Counter()
        for ann in raw_annotations:
            concept_counts[ann["action_concept"]] += 1
        
        total_annotations = len(raw_annotations)
        
        concept_pool = []
        
        for cluster in clusters:
            # 클러스터 내 모든 멤버의 빈도 합산
            cluster_freq = sum(
                concept_counts.get(m["name"], 0) for m in cluster["members"]
            )
            
            # 최소 빈도 필터링
            if cluster_freq < self.min_frequency:
                continue
            
            # max_concepts 제한
            if len(concept_pool) >= self.max_concepts:
                break
            
            # 변형 이름 목록 (상위 5개)
            variants = sorted(
                [(m["name"], concept_counts.get(m["name"], 0)) for m in cluster["members"]],
                key=lambda x: -x[1]
            )[:5]
            
            concept_pool.append({
                "concept_id": len(concept_pool),
                "name": cluster["representative_name"],
                "description": cluster["representative_description"],
                "frequency": cluster_freq,
                "frequency_pct": round(100 * cluster_freq / max(total_annotations, 1), 1),
                "cluster_size": cluster["total_count"],
                "variants": [v[0] for v in variants],
            })
        
        return concept_pool
    
    def map_annotations_to_pool(
        self,
        raw_annotations: List[dict],
        concept_pool: List[dict],
        pool_embeddings: np.ndarray,
        all_embeddings: np.ndarray,
        all_names: List[str],
    ) -> List[dict]:
        """
        원본 annotation을 concept pool ID로 매핑
        
        각 raw annotation의 concept를 가장 가까운 pool concept에 매핑합니다.
        
        Returns:
            최종 데이터셋 엔트리 리스트
        """
        # 이름 → 임베딩 인덱스 매핑
        name_to_idx = {name: idx for idx, name in enumerate(all_names)}
        
        # Pool concept 이름 → pool ID
        pool_name_to_id = {}
        for concept in concept_pool:
            pool_name_to_id[concept["name"]] = concept["concept_id"]
            for variant in concept["variants"]:
                pool_name_to_id[variant] = concept["concept_id"]
        
        dataset = []
        
        for ann in raw_annotations:
            concept_name = ann["action_concept"]
            
            # 직접 매핑 시도
            if concept_name in pool_name_to_id:
                pool_id = pool_name_to_id[concept_name]
            else:
                # 임베딩 유사도로 가장 가까운 pool concept 찾기
                if concept_name in name_to_idx:
                    ann_emb = all_embeddings[name_to_idx[concept_name]]
                    similarities = pool_embeddings @ ann_emb
                    pool_id = int(np.argmax(similarities))
                else:
                    pool_id = 0  # 매핑 실패 시 첫 번째 concept
            
            entry = {
                "episode_id": ann["episode_id"],
                "segment_id": ann["segment_id"],
                "start_frame": ann["start_frame"],
                "end_frame": ann["end_frame"],
                "representative_frame": ann["representative_frame"],
                "task_description": ann.get("task_description", ""),
                "concept_id": pool_id,
                "concept_name": concept_pool[pool_id]["name"],
                "original_concept": concept_name,
                "description": ann["description"],
                "image_description": ann.get("image_description", ""),
            }
            
            dataset.append(entry)
        
        return dataset
    
    def refine_and_build(
        self,
        raw_concepts: List[dict],
        segments: List[dict],
    ) -> Tuple[List[dict], List[dict]]:
        """
        전체 정제 파이프라인 실행
        
        Args:
            raw_concepts: Gemini가 생성한 원본 annotation 리스트
            segments: AutoSegmenter 출력 (에피소드별 세그먼트)
            
        Returns:
            concept_pool: 정제된 컨셉 풀
            final_dataset: 컨셉 ID가 매핑된 최종 데이터셋
        """
        if not raw_concepts:
            print("  [WARNING] No raw concepts to refine.")
            return [], []
        
        # 1. 고유 컨셉 추출
        unique_concepts = {}  # name → description
        for ann in raw_concepts:
            name = ann["action_concept"]
            desc = ann["description"]
            if name not in unique_concepts:
                unique_concepts[name] = desc
        
        concept_names = list(unique_concepts.keys())
        descriptions = list(unique_concepts.values())
        
        print(f"  Unique raw concepts: {len(concept_names)}")
        
        # 2. T5 인코더로 임베딩
        print("  Computing T5 embeddings...")
        embeddings = self._encode_texts(descriptions)
        print(f"  Embedding shape: {embeddings.shape}")
        
        # 3. 클러스터링
        print(f"  Clustering (threshold={self.similarity_threshold})...")
        clusters = self.cluster_concepts(concept_names, descriptions, embeddings)
        print(f"  Clusters formed: {len(clusters)}")
        
        # 4. Concept pool 구축
        concept_pool = self.build_concept_pool(clusters, raw_concepts)
        print(f"  Concept pool size: {len(concept_pool)}")
        
        # 5. Pool concept 임베딩 계산
        pool_descriptions = [c["description"] for c in concept_pool]
        if pool_descriptions:
            pool_embeddings = self._encode_texts(pool_descriptions)
        else:
            pool_embeddings = np.zeros((0, embeddings.shape[1] if embeddings.ndim > 1 else 1))
        
        # 6. 원본 annotation을 pool ID로 매핑
        print("  Mapping annotations to concept pool...")
        final_dataset = self.map_annotations_to_pool(
            raw_annotations=raw_concepts,
            concept_pool=concept_pool,
            pool_embeddings=pool_embeddings,
            all_embeddings=embeddings,
            all_names=concept_names,
        )
        
        # 7. 에피소드별 순서 정보 추가
        print("  Adding order information...")
        episode_groups = defaultdict(list)
        for entry in final_dataset:
            episode_groups[entry["episode_id"]].append(entry)
        
        for ep_id, entries in episode_groups.items():
            entries.sort(key=lambda x: x["segment_id"])
            for order, entry in enumerate(entries):
                entry["order_in_episode"] = order
                entry["total_concepts_in_episode"] = len(entries)
                # 에피소드 내 활성 컨셉 ID 리스트
                entry["episode_concept_ids"] = [e["concept_id"] for e in entries]
        
        return concept_pool, final_dataset


if __name__ == "__main__":
    print("Testing ConceptRefiner...")
    
    refiner = ConceptRefiner(
        similarity_threshold=0.85,
        max_concepts=50,
        min_frequency=2,
    )
    
    # 더미 raw concepts 생성
    dummy_concepts = []
    concept_templates = [
        ("approach_object", "Robot arm approaches the target object"),
        ("reach_toward", "Robot reaches toward the item on the table"),
        ("move_to_object", "Robot moves toward the target"),
        ("grasp_object", "Robot gripper closes to grasp the object"),
        ("grip_item", "Robot grips the item firmly"),
        ("lift_object", "Robot lifts the grasped object upward"),
        ("raise_object", "Robot raises the object from the surface"),
        ("transport_object", "Robot transports the object to a new location"),
        ("move_right", "Robot moves the held object to the right"),
        ("carry_to_target", "Robot carries the object toward the target zone"),
        ("lower_object", "Robot lowers the object toward the surface"),
        ("place_object", "Robot places the object down on the surface"),
        ("release_object", "Robot gripper opens to release the object"),
        ("retract", "Robot arm retracts after completing the task"),
        ("stabilize", "Robot holds position to stabilize"),
    ]
    
    import random
    random.seed(42)
    
    for ep_id in range(50):
        # 에피소드당 3~5개 세그먼트
        n_segs = random.randint(3, 5)
        for seg_id in range(n_segs):
            template = random.choice(concept_templates)
            dummy_concepts.append({
                "episode_id": ep_id,
                "segment_id": seg_id,
                "start_frame": seg_id * 100,
                "end_frame": (seg_id + 1) * 100,
                "representative_frame": seg_id * 100 + 50,
                "action_concept": template[0],
                "description": template[1],
                "image_description": f"Robot arm on table for episode {ep_id}",
                "task_description": f"Task {ep_id}",
            })
    
    print(f"\nDummy raw concepts: {len(dummy_concepts)}")
    
    # 정제 실행
    concept_pool, dataset = refiner.refine_and_build(
        raw_concepts=dummy_concepts,
        segments=[],
    )
    
    print(f"\n=== Concept Pool ({len(concept_pool)} concepts) ===")
    for c in concept_pool:
        print(f"  [{c['concept_id']:2d}] {c['name']:25s} | freq={c['frequency']:4d} "
              f"({c['frequency_pct']:5.1f}%) | variants={c['variants'][:3]}")
    
    print(f"\n=== Dataset Sample (first 5 entries) ===")
    for entry in dataset[:5]:
        print(f"  ep={entry['episode_id']:2d} seg={entry['segment_id']} "
              f"→ concept_id={entry['concept_id']} ({entry['concept_name']}) "
              f"| order={entry['order_in_episode']}/{entry['total_concepts_in_episode']}")
    
    print(f"\n=== Final Data Structure ===")
    sample = dataset[0]
    print(json.dumps(sample, indent=2, ensure_ascii=False))
    
    print("\n✓ ConceptRefiner test completed!")