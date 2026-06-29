# cashlog-auto

cashlog 앱용 상품 인식 및 지출 카테고리 추천 모델 학습/추론 리포지토리.

사진(영수증 또는 상품)에서 주요 상품을 찾아 고정 카테고리로 분류하고, 앱에 카테고리를 추천한다. 타겟은 iOS/Android 폰(온디바이스 우선, 서버 폴백 허용)이며 M1 Pro는 개발/학습용이다.

## 전략 요약

라벨 데이터가 없는 상태에서 시작하므로 단계적으로 접근한다.

1. 학습 없이 동작하는 파운데이션 모델로 베이스라인 구성 (CLIP zero-shot, OCR)
2. Qwen2.5-VL로 데이터 자동 라벨링 + 휴먼 검수 (부트스트래핑)
3. 부트스트랩 데이터로 폰용 경량 모델 증류/학습
4. Core ML(iOS) + TFLite(Android) export, 양자화, M1 Pro 벤치마크

## 파이프라인

```
사진 ─▶ 라우터(영수증/상품)
        ├─ 영수증 ─▶ OCR ─▶ 품목 파싱 ─▶ 텍스트 카테고리 분류
        └─ 상품   ─▶ CLIP 임베딩 ─▶ 카테고리 매칭
                          │
                  신뢰도 낮으면 ─▶ 서버 Qwen2.5-VL 폴백
```

- 입력 두 종류를 분리 트랙으로 처리하고, 공통 카테고리 체계([data/categories.json](data/categories.json))를 공유한다.
- 온디바이스 결과 신뢰도가 임계값 미만이면 서버 VLM으로 폴백한다.

## 구조

```
src/cashlog/
  categories.py        # 고정 카테고리 체계 로딩
  pipeline.py          # end-to-end 오케스트레이션 (+ VLM 폴백)
  router.py            # 영수증 vs 상품 라우팅
  product/             # 상품 사진 트랙 (CLIP zero-shot / 학습 헤드 / export)
  receipt/             # 영수증 트랙 (OCR / 파싱 / 텍스트 분류)
  vlm/qwen_vl.py       # Qwen2.5-VL 래퍼 (폴백 + 자동 라벨링)
  dataset.py           # 라벨 저장소(JSONL) 입출력
scripts/               # 부트스트래핑 / 학습 / export / 벤치마크
model/qwen.25/         # VLM 로딩 노트북
data/categories.json   # 카테고리 체계 (앱에 맞게 수정)
```

## 설치

```bash
pip install -e .            # 기본 (추론/zero-shot)
pip install -e .[vlm]       # Qwen2.5-VL 폴백/라벨링
pip install -e .[ocr]       # easyocr (영수증 OCR)
pip install -e .[train]     # 경량 모델 학습 (scikit-learn 등)
pip install -e .[export]    # Core ML / ONNX export
pip install -e .[reviews]   # 앱 리뷰 크롤링 + 엑셀 export
```

Python 3.11 기준 (Apple Silicon은 자동으로 MPS 사용).

## 사용

```bash
# 단일 이미지 데모
python main.py path/to/image.jpg --type product

# Phase 2: 자동 라벨링 -> 검수 -> 분할
python scripts/bootstrap_label.py          # data/raw/ 이미지 -> data/labeled/labels.jsonl
#   (라벨 검수 후 각 레코드의 "reviewed"를 true로)
python scripts/split_dataset.py

# Phase 3: 경량 모델 학습
python scripts/train_product.py            # CLIP 헤드 (상품)
python scripts/train_receipt.py            # 텍스트 분류기 (영수증)

# Phase 4: export + 벤치마크
python scripts/export_coreml.py --quantize int8
python scripts/export_tflite.py
python scripts/benchmark.py --mode zeroshot
```

## 앱 리뷰 크롤링

앱 이름만 넘기면 Play Store/App Store에서 앱을 검색하고, 두 스토어의 리뷰를 합쳐 엑셀 파일로 저장한다.

```bash
python scripts/export_app_reviews.py adidas --count 500
```

직접 스토어 ID를 지정할 수도 있다.

```bash
python scripts/export_app_reviews.py \
  --playstore-app-id com.adidas.app \
  --appstore-app-name adidas \
  --country kr \
  --lang ko \
  --count 500 \
  --out data/reviews/adidas_reviews.xlsx
```

- Play Store만 받을 때: `--playstore-app-id`만 전달
- App Store만 받을 때: `--appstore-app-name`만 전달
- 이름 검색으로 한쪽 스토어만 받을 때: `--store playstore` 또는 `--store appstore` 전달
- App Store 검색이 부정확하면 숫자 앱 ID를 `--appstore-app-id`로 같이 전달
- 엑셀 컬럼: `source`, `userName`, `review`, `score`, `date`, `title`, `appVersion`, `reviewId`, `thumbsUpCount`, `developerReply`, `repliedAt`

## 카테고리 수정

[data/categories.json](data/categories.json)에서 실제 앱 카테고리에 맞게 편집한다.
- `aliases`: CLIP zero-shot 프롬프트(영문 권장)
- `keywords`: 영수증 OCR 한글 키워드 매칭

## 한국어 성능 참고

기본 CLIP(`openai/clip-vit-base-patch32`)은 영어 중심이다. 한국어 프롬프트/온디바이스 배포 시 MobileCLIP 또는 다국어 CLIP 체크포인트로 [src/cashlog/config.py](src/cashlog/config.py)의 `DEFAULT_CLIP_MODEL`을 교체한다. 파이프라인의 나머지는 동일하다.
```
