# review_bot

앱 이름으로 Play Store와 App Store 리뷰를 수집하고 엑셀 파일로 내려받는 Vercel 앱입니다.

## 기능

- 앱 이름 검색으로 Play Store 패키지 ID 자동 확인
- App Store 앱 이름 검색
- Play Store/App Store 리뷰 통합 수집
- `.xlsx` 다운로드
- Vercel 정적 페이지 + Python API 함수로 배포

## 배포 구조

```
index.html              # 리뷰 다운로드 화면
api/reviews.py          # Vercel Python API
src/cashlog/reviews.py  # 리뷰 수집 및 엑셀 생성 로직
scripts/export_app_reviews.py  # 로컬 CLI 실행용
requirements.txt        # Vercel 설치 의존성
vercel.json             # Vercel 라우팅/설치 설정
```

## 로컬 실행

```bash
pip install -r requirements.txt
python scripts/export_app_reviews.py adidas --count 100
```

기본 출력 파일은 `data/reviews/{앱이름}_reviews.xlsx`입니다.

## 웹 사용

Vercel 배포 후 루트 페이지에서 앱 이름을 입력하고 `엑셀 다운로드`를 누르면 됩니다.

```text
https://review-bot-6hv5.vercel.app
```

## CLI 옵션

```bash
python scripts/export_app_reviews.py adidas \
  --store both \
  --country kr \
  --lang ko \
  --count 500 \
  --out data/reviews/adidas_reviews.xlsx
```

- `--store both`: Play Store와 App Store 모두 수집
- `--store playstore`: Play Store만 수집
- `--store appstore`: App Store만 수집
- `--count`: 스토어별 리뷰 수
- `--score`: Play Store 별점 필터

## Node CLI: 키워드 검색 상위 앱 리뷰 수집

검색 키워드로 Google Play와 App Store의 상위 앱을 찾고, 각 앱의 리뷰를 Excel 파일로 저장하는 Node CLI도 포함되어 있습니다.

```bash
npm install
npm start -- --keyword "가계부" --country kr --lang ko --top 10 --reviews 50
```

기본 출력 파일은 `output/reviews_<keyword>_<timestamp>.xlsx`입니다.

주요 옵션:

```text
-k, --keyword <text>   검색 키워드. 필수.
    --country <code>   스토어 국가 코드. 기본값: kr
    --lang <code>      Google Play 언어 코드. 기본값: ko
    --top <number>     스토어별 검색 상위 앱 개수. 기본값: 10
    --reviews <number> 앱당 리뷰 개수. 기본값: 50
    --store <name>     google, apple, both 중 하나. 기본값: both
    --out <file>       출력 .xlsx 경로.
```

Excel 파일에는 `Summary`, `Apps`, `Reviews`, 오류가 있을 경우 `Errors` 시트가 생성됩니다.

## GitHub 정리 방식

리뷰 봇과 무관한 기존 ML/모델/데이터 파일은 로컬에 남기되 `.gitignore`와 `git rm --cached`로 Git 추적에서 제외했습니다.
