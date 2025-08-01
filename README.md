# 잡코리아 AI Challenge - LLM Engineer

## 프로젝트 개요
채용공고 자동생성 GenAI 서비스의 할루시네이션 문제와 응답 지연 문제를 해결하는 위상정보 기반 매칭 시스템

### 핵심 솔루션
- **위상정보 시스템**: 기업-구직자 간 관계성 기반 매칭
- **가중치 기반 임베딩**: 정량적 지표를 활용한 객관적 매칭
- **하이브리드 임베딩**: 클라이언트-서버 분산 처리
- **RAG + 위상경계**: 환각 현상 원천 차단

## 빠른 시작

### 1. 자동 설정 (권장)
```bash
# 환경 자동 설정
python setup_env.py

# 가상환경 활성화
activate.bat  # Windows
source activate.sh  # Linux/Mac

# 테스트 실행
python test_final.py
```

### 2. 수동 설정
```bash
# 가상환경 생성
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements-minimal.txt  # 최소 설치
# pip install -r requirements.txt  # 전체 설치

# 환경 변수 설정 (선택사항)
cp .env.example .env
```

## 테스트 가이드

```bash
# 1. 빠른 동작 확인
python test_simple.py

# 2. 전체 시스템 테스트 (Mock 임베딩)
python test_final.py

# 3. 기능 데모
python demo_showcase.py

# 4. 성능 테스트
python scripts/performance_test.py

# 5. API 서버 실행
python scripts/run_server.py
```

## 문제 해결

### memory_profiler 오류
```bash
pip install -r requirements-minimal.txt
```

### sentence-transformers 오류
- test_final.py 사용 (자동으로 Mock 임베딩 사용)
- 또는 실제 설치: `pip install sentence-transformers==2.2.2`

### 한글 깨짐 (Windows)
```bash
chcp 65001
```

## 주요 성능 지표
- 할루시네이션 비율: < 3%
- 응답 시간: < 1초 (캐시), < 3초 (신규)
- 매칭 정확도: > 90%
- 서버 비용 절감: > 70%