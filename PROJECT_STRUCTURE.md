# 프로젝트 구조

## 디렉토리 구조
```
jobkorea-ai-challenge/
│
├── config/                 # 설정 파일
│   ├── settings.py        # 애플리케이션 설정
│   └── prompts.yaml       # 프롬프트 템플릿
│
├── data/                  # 데이터 디렉토리
│   ├── companies/         # 회사 데이터
│   ├── job_seekers/       # 구직자 데이터
│   ├── embeddings/        # 임베딩 저장소
│   ├── demo_companies.json    # 데모 회사 데이터
│   └── demo_candidates.json   # 데모 구직자 데이터
│
├── src/                   # 소스 코드
│   ├── embeddings/        # 임베딩 시스템
│   │   ├── base_embedder.py
│   │   ├── company_embedder.py
│   │   └── candidate_embedder.py
│   │
│   ├── topology/          # 위상정보 시스템
│   │   ├── topology_mapper.py
│   │   ├── gravity_field.py
│   │   └── boundary_validator.py
│   │
│   ├── rag/              # RAG 시스템
│   │   ├── vector_store.py
│   │   └── retriever.py
│   │
│   ├── matching/         # 매칭 시스템
│   │   ├── weighted_matcher.py
│   │   └── bidirectional_optimizer.py
│   │
│   ├── generation/       # 생성 시스템
│   │   ├── llm_generator.py
│   │   └── hallucination_guard.py
│   │
│   ├── validation/       # 검증 시스템
│   │   ├── fact_checker.py
│   │   └── consistency_validator.py
│   │
│   └── api/             # API 서버
│       ├── server.py
│       └── client_handler.py
│
├── scripts/             # 실행 스크립트
│   ├── demo.py         # 기능 데모
│   ├── performance_test.py  # 성능 테스트
│   └── run_server.py   # API 서버 실행
│
├── tests/              # 단위 테스트
│   ├── test_embeddings.py
│   ├── test_topology.py
│   └── test_generation.py
│
├── docs/               # 문서
├── notebooks/          # 주피터 노트북
│
├── test_simple.py      # 간단한 동작 확인
├── test_final.py       # 통합 시스템 테스트 (권장)
├── test_direct.py      # 직접 모듈 테스트
├── demo_showcase.py    # 기능 시연 데모
├── setup_env.py        # 환경 자동 설정
│
├── requirements.txt          # 전체 의존성
├── requirements-minimal.txt  # 최소 의존성
├── README.md          # 프로젝트 설명
├── INSTALL.md         # 설치 가이드
├── DEPLOY.md          # 배포 가이드
├── TEST_RESULTS.md    # 테스트 결과
└── TEST_GUIDE.md      # 테스트 가이드
```

## 핵심 모듈 설명

### 1. Embeddings (임베딩)
- 텍스트와 수치 데이터를 결합한 하이브리드 임베딩
- 회사/구직자별 특화된 특성 추출

### 2. Topology (위상정보)
- 그래프 기반 관계성 모델링
- 중력장 시뮬레이션으로 자연스러운 클러스터링

### 3. RAG (검색 증강 생성)
- FAISS 기반 고속 벡터 검색
- 위상정보 활용한 관련성 높은 검색

### 4. Matching (매칭)
- 다차원 가중치 기반 매칭
- 양방향 최적화 알고리즘

### 5. Generation (생성)
- LLM 기반 콘텐츠 생성
- 다층 환각 방지 시스템

### 6. Validation (검증)
- 팩트 체킹
- 일관성 검증

## 삭제된 파일
- ~~test_simple_mock.py~~ (test_direct.py로 통합)
- ~~test_robust.py~~ (test_final.py로 개선)
- ~~install_dependencies.py~~ (setup_env.py로 대체)
- ~~nul~~ (실수로 생성된 파일)