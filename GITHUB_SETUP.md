# GitHub 저장소 설정 가이드

## 1. GitHub에서 새 저장소 생성

1. GitHub.com에 로그인
2. 우측 상단 '+' 버튼 클릭 → 'New repository' 선택
3. 저장소 정보 입력:
   - Repository name: `jobkorea-ai-challenge`
   - Description: "위상정보 기반 채용 매칭 시스템 - JobKorea AI Challenge"
   - Public/Private 선택
   - **주의**: "Initialize this repository with..." 옵션들은 모두 체크 해제 (이미 로컬에 파일이 있으므로)

## 2. 로컬 저장소를 GitHub에 연결

저장소 생성 후 나타나는 명령어를 실행하거나, 아래 명령어 사용:

```bash
# GitHub 저장소 URL로 변경 (예: https://github.com/YOUR_USERNAME/jobkorea-ai-challenge.git)
git remote add origin https://github.com/YOUR_USERNAME/jobkorea-ai-challenge.git

# 브랜치 이름을 main으로 변경 (선택사항)
git branch -M main

# 첫 푸시
git push -u origin main
```

## 3. SSH 키 사용 (선택사항)

HTTPS 대신 SSH를 사용하려면:

```bash
# SSH URL로 변경
git remote set-url origin git@github.com:YOUR_USERNAME/jobkorea-ai-challenge.git
```

## 4. 저장소 설정 확인

```bash
# 원격 저장소 확인
git remote -v

# 현재 상태 확인
git status
```

## 5. GitHub Actions 설정 (선택사항)

자동 테스트를 위한 GitHub Actions 워크플로우:

`.github/workflows/test.yml` 파일 생성:

```yaml
name: Tests

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements-minimal.txt
        
    - name: Run tests
      run: |
        python test_final.py
```

## 6. README 뱃지 추가 (선택사항)

README.md 상단에 추가:

```markdown
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Tests](https://github.com/YOUR_USERNAME/jobkorea-ai-challenge/workflows/Tests/badge.svg)
```

## 7. 저장소 설정 완료 후

1. **Settings** → **About** 섹션에서 설명과 태그 추가
2. 주요 태그: `ai`, `machine-learning`, `nlp`, `job-matching`, `korean`, `rag`, `embeddings`
3. **Issues** 탭 활성화하여 피드백 받기
4. **Wiki** 탭에서 상세 문서 작성 (선택사항)

## 문제 해결

### Permission denied 오류
```bash
# HTTPS로 변경
git remote set-url origin https://github.com/YOUR_USERNAME/jobkorea-ai-challenge.git

# 또는 Personal Access Token 사용
# GitHub Settings → Developer settings → Personal access tokens
```

### Large file 오류
```bash
# Git LFS 설치 (대용량 파일용)
git lfs track "*.pkl"
git lfs track "*.npy"
git add .gitattributes
```