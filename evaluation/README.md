# PhoMate 검색 성능 평가 하네스

Phomate 사진 검색의 검색 방식별 성능을 정량 평가합니다.  
프로덕션 컬렉션(`post_vectors_siglip2`)은 건드리지 않고, 별도 평가 컬렉션을 사용합니다.

## 평가 방식

| ID | 방식 | 구현 상태 |
|---|---|---|
| A | SigLIP 이미지 임베딩, 한국어 쿼리 그대로 | ✅ 완전 구현 |
| A' | SigLIP, 영어 번역 쿼리 | 🔧 TODO (VLM_API_KEY 필요) |
| B | VLM 캡션 → 텍스트 임베딩 검색 | 🔧 TODO |
| B+C | B + concept IDF 가중 재랭킹 | 🔧 TODO |

## 사전 조건

- Qdrant 서버 실행 중 (`QDRANT_URL` 환경변수로 지정)
- Python 3.10+
- `embedding_worker/requirements.txt` 설치 완료 (torch, transformers 포함)

## 설치

```bash
cd PhoMate_FastAPI_BE
pip install -r embedding_worker/requirements.txt
pip install -r evaluation/requirements.txt
```

## 환경변수 설정 (.env)

```bash
# PhoMate_FastAPI_BE/.env
QDRANT_URL=http://localhost:6333

# A'/B/B+C 방식 사용 시
VLM_API_KEY=sk-...
VLM_MODEL=gpt-4o-mini
TEXT_EMBED_MODEL=text-embedding-3-small
```

## 실행 순서

모든 커맨드는 `PhoMate_FastAPI_BE/` 에서 실행합니다.

### 1. 이미지 인덱싱 (A / A' 방식)

```bash
python -m evaluation.prepare_corpus --dataset-dir /path/to/images
# --limit 100  # 테스트용 소규모 실행
```

→ `evaluation/data/id_map.csv` 생성, `eval_siglip2` 컬렉션 인덱싱

### 2. 쿼리 준비

```bash
# known_item 쿼리 자동 생성 (doc_id 기반)
python -m evaluation.build_queries --auto

# 쿼리 수동 추가
python -m evaluation.build_queries --add "카페에서 커피 마시는 사람" broad
```

→ `evaluation/data/queries.csv` 생성·편집

### 3. 정답(qrels) 생성

```bash
python -m evaluation.make_qrels --mode known_item
```

→ `evaluation/data/qrels.txt` 생성

### 4. 검색 실행

```bash
# 방식 A만 (현재 완전 구현된 방식)
python -m evaluation.run_search --methods A

# 전체 방식 (A'/B/B+C는 TODO 구현 후)
python -m evaluation.run_search
```

→ `evaluation/data/runs/<method>.txt` 생성

### 5. 지표 산출

```bash
python -m evaluation.evaluate
```

→ 콘솔 비교 테이블 출력  
→ `evaluation/data/results_summary.csv`, `results_per_query.csv` 생성

## 테스트 실행

```bash
cd PhoMate_FastAPI_BE
pytest evaluation/tests/ -v
```

모델·DB 없이 순수 함수만 테스트합니다 (TREC I/O, IDF, concept overlap).

## 데이터 디렉토리 구조

```
evaluation/data/
  id_map.csv            # doc_id ↔ 파일 경로 ↔ Qdrant ID
  queries.csv           # 평가 쿼리 목록
  qrels.txt             # 정답 (TREC 포맷)
  concept_vocab.csv     # concept_id ↔ 레이블 (prepare_concepts.py 출력)
  photo_concepts.csv    # 이미지별 concept 목록
  runs/
    A.txt               # TREC run 파일 (방식별)
    A_prime.txt
    B.txt
    BC.txt
  results_summary.csv   # 방식별 집계 지표
  results_per_query.csv # 쿼리별 세부 지표
```

## 평가 지표

| 지표 | 설명 |
|---|---|
| recall@K | 정답 중 상위 K개 안에 포함된 비율 |
| precision@K | 상위 K개 중 정답 비율 |
| mrr@10 | Mean Reciprocal Rank (첫 번째 정답 순위의 역수 평균) |
| ndcg@10 | Normalized Discounted Cumulative Gain |
