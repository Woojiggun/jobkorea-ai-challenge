# 배포 가이드

## 시스템 요구사항

- Python 3.8+
- 8GB+ RAM (추천: 16GB)
- GPU (선택사항, 임베딩 가속화용)

## 설치 방법

### 1. 환경 설정

```bash
# 저장소 클론
git clone https://github.com/your-repo/jobkorea-ai-challenge.git
cd jobkorea-ai-challenge

# 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

### 2. 환경 변수 설정

```bash
# .env 파일 생성
cp .env.example .env

# .env 파일 편집하여 API 키 설정
# OPENAI_API_KEY=your_api_key_here
```

### 3. 데이터 준비

```bash
# 데모 데이터로 시작
python scripts/demo.py

# 또는 실제 데이터 임포트
python scripts/import_data.py --companies data/companies.json --candidates data/candidates.json
```

## 실행 방법

### API 서버 실행

```bash
# 개발 모드
python run_server.py

# 프로덕션 모드
gunicorn src.api.server:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### 성능 테스트

```bash
python scripts/performance_test.py
```

## Docker 배포

### Dockerfile

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["gunicorn", "src.api.server:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]
```

### Docker Compose

```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
```

## 프로덕션 최적화

### 1. 임베딩 캐싱

```python
# 사전 임베딩 생성
python scripts/precompute_embeddings.py

# Redis 캐시 설정
REDIS_URL=redis://localhost:6379
```

### 2. 로드 밸런싱

```nginx
upstream app {
    server app1:8000;
    server app2:8000;
    server app3:8000;
}

server {
    listen 80;
    location / {
        proxy_pass http://app;
    }
}
```

### 3. 모니터링

- Prometheus + Grafana 설정
- 로그 수집 (ELK Stack)
- 성능 메트릭 추적

## 보안 고려사항

1. API 키 환경 변수로 관리
2. HTTPS 사용
3. Rate limiting 적용
4. 입력 검증 강화

## 문제 해결

### 메모리 부족

```bash
# 배치 크기 조정
BATCH_SIZE=16

# 경량 임베딩 모델 사용
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

### 응답 지연

- Redis 캐시 활성화
- 임베딩 사전 계산
- 비동기 처리 활용

## 지원

문제 발생 시:
- GitHub Issues: https://github.com/your-repo/issues
- Email: support@example.com