# 테스트 가이드

## 테스트 파일 설명

### 1. test_final.py (메인 테스트) ✅
- **용도**: 전체 시스템 통합 테스트
- **특징**: Mock 임베딩 사용, 의존성 최소화
- **실행**: `python test_final.py`
- **추천**: 시스템 전체 동작 확인 시 사용

### 2. test_direct.py (직접 테스트) ✅
- **용도**: 설정 파일 우회하여 핵심 기능만 테스트
- **특징**: pydantic 의존성 제거, 단위 기능 검증
- **실행**: `python test_direct.py`
- **추천**: 개별 모듈 동작 확인 시 사용

### 3. test_simple.py (간단 테스트) ✅
- **용도**: 임포트 및 기본 기능 테스트
- **특징**: 최소한의 테스트로 빠른 확인
- **실행**: `python test_simple.py`
- **추천**: 설치 후 첫 확인 시 사용

### 4. demo_showcase.py (데모) ✅
- **용도**: 주요 기능 시연 및 성능 측정
- **특징**: 실제 사용 시나리오 시뮬레이션
- **실행**: `python demo_showcase.py`
- **추천**: 기능 데모 시 사용

### 5. tests/ 디렉토리 (단위 테스트)
- **용도**: pytest 기반 단위 테스트
- **실행**: `pytest tests/`
- **참고**: 전체 의존성 설치 필요

## 삭제된 파일
- ~~test_simple_mock.py~~ → test_direct.py로 통합
- ~~test_robust.py~~ → test_final.py로 개선
- ~~nul~~ → 실수로 생성된 파일

## 권장 테스트 순서

1. **빠른 확인**
   ```bash
   python test_simple.py
   ```

2. **전체 시스템 테스트**
   ```bash
   python test_final.py
   ```

3. **기능 데모**
   ```bash
   python demo_showcase.py
   ```

4. **성능 테스트** (전체 설치 필요)
   ```bash
   python scripts/performance_test.py
   ```

## 문제 해결

### "No module named 'sentence_transformers'" 오류
- test_final.py 또는 test_direct.py 사용 (Mock 임베딩 자동 사용)

### Unicode/한글 깨짐
- Windows: `chcp 65001` 실행 후 테스트
- 또는 test_final.py 사용 (영문 출력)

### pydantic_settings 오류
- test_direct.py 사용 (설정 모듈 우회)