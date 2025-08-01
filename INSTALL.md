# 설치 가이드

## 빠른 시작

### 1. 최소 설치 (memory_profiler, matplotlib 제외)

```bash
# 프로젝트 디렉토리로 이동
cd jobkorea-ai-challenge

# 가상환경 생성 및 활성화
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# 최소 의존성 설치
pip install -r requirements-minimal.txt
```

### 2. 전체 설치 (모든 기능 포함)

```bash
# 전체 의존성 설치
pip install -r requirements.txt
```

## 문제 해결

### memory_profiler 오류

memory_profiler가 없어도 기본 기능은 모두 동작합니다.
성능 테스트에서 메모리 프로파일링만 제외됩니다.

```bash
# memory_profiler 없이 실행
pip install -r requirements-minimal.txt
```

### matplotlib 오류

matplotlib가 없으면 성능 그래프만 생성되지 않습니다.
텍스트 결과는 정상적으로 출력됩니다.

### Import 오류

프로젝트 루트 디렉토리에서 실행하세요:

```bash
# 올바른 실행 방법
cd jobkorea-ai-challenge
python scripts/demo.py

# 잘못된 실행 방법
cd scripts
python demo.py  # 이렇게 하면 import 오류 발생
```

## 테스트 실행

### 1. 간단한 테스트

```bash
python test_simple.py
```

### 2. 데모 실행

```bash
python scripts/demo.py
```

### 3. 성능 테스트

```bash
python scripts/performance_test.py
```

### 4. 단위 테스트

```bash
pytest tests/
```

## 환경 변수 설정

OpenAI API를 사용하는 경우:

```bash
# .env 파일 생성
copy .env.example .env

# .env 파일 편집
# OPENAI_API_KEY=your_api_key_here
```

## 최소 시스템 요구사항

- Python 3.8 이상
- 4GB RAM (추천: 8GB)
- 2GB 디스크 공간