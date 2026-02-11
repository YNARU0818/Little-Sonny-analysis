# 🚗 Used Car Price Prediction (발표/포트폴리오용 README)

> 중고차의 제원·상태 정보를 활용해 **차량 가격(`price`)을 예측**하는 회귀(Regression) 프로젝트입니다.  
> 본 레포지토리는 `used_car.ipynb`를 중심으로 **데이터 이해 → EDA 인사이트 → Feature Engineering → 모델링(LightGBM/CatBoost) → 교차검증(CV) → 제출파일 생성** 흐름을 담고 있습니다.

---

## 0. 발표 흐름(Agenda)

1. **문제 정의 & 평가 지표**
2. **데이터셋 이해(컬럼/결측/분포)**
3. **EDA 핵심 인사이트**
4. **Feature Engineering 설계 의도**
5. **모델링 전략(Leak-free CV)**
6. **결과(성능) & 해석**
7. **에러/한계 & 개선 방향**
8. **재현 방법(How to Run) & 제출(submission) 생성**

---

## 1. Problem Statement

- 목표: `train.csv`의 학습 데이터를 이용해 `test.csv`의 **`price`를 예측**
- 문제 유형: **회귀(Regression)**
- 타깃 분포: 가격은 일반적으로 **long-tail(긴 꼬리)** 분포를 가지므로, 노트북에서는 학습 안정화를 위해 `log1p(price)`로 변환해 학습합니다.

### Evaluation Metric (노트북 기준)
- **RMSE on `log1p(price)`**
- KFold 5-split (seed=42)로 평균 성능을 측정합니다.

---

## 2. Dataset Overview

노트북 상단의 데이터 설명 셀을 기반으로 요약했습니다.

- 각 row = **차량 1대**
- 규모: **188,533 rows / 13 features (+ target price)**

### Columns

| 컬럼 | 타입 | 설명 |
|---|---|---|
| `id` | int | 차량 고유 식별자 |
| `brand` | cat | 제조사 |
| `model` | cat | 모델명 |
| `model_year` | num | 연식(출고 연도) |
| `milage` | num | 누적 주행거리 |
| `fuel_type` | cat | 연료 유형 |
| `engine` | cat/text | 엔진 사양 문자열(HP, 배기량, 실린더 등 포함) |
| `transmission` | cat | 변속기 |
| `ext_col` | cat | 외장 색상 |
| `int_col` | cat | 내장 색상 |
| `accident` | cat | 사고/손상 이력 |
| `clean_title` | cat | 타이틀(소유권) 상태 |
| `price` | num | (train only) 타깃 가격 |

---

## 3. EDA 핵심 요약

### 3.1 결측치(Missing Values)

노트북 출력 기준 결측 개수:

- `fuel_type`: 5,083
- `accident`: 2,452
- `clean_title`: 21,419  ← **가장 큼**

> 특히 `clean_title` 결측 비중이 꽤 커서, “정보 미제공” 자체가 신호일 수 있다고 보고 **`MISSING` 카테고리로 보존**했습니다.

---

### 3.2 수치형 분포(구간별 비율)

#### `model_year` (연식)
노트북의 binning 결과(대략 균등 분할):

- 2018~2020 구간이 **16.58%**로 가장 높고,
- 2014~2016이 **13.31%**,
- 2022~2024는 **4.74%**로 상대적으로 적습니다.

#### `milage` (주행거리)
- 전반적으로 구간별 비중이 **약 10% 내외로 고르게 분포**하도록 binning 되어 있으며,
- 고주행(최상위 구간 136,731~405,000)도 **약 9.98%**로 존재합니다.

> 결론: 특정 구간에 심하게 쏠리기보다는, 연식/주행거리 모두 다양한 범위를 커버합니다.

---

### 3.3 범주형 분포(Top 브랜드)

`brand` 분포 상위(노트북 출력 기준):

1. Ford (12.25%)
2. Mercedes-Benz (10.17%)
3. BMW (9.03%)
4. Chevrolet (8.66%)
5. Audi (5.77%)
6. Porsche (5.63%)
7. Land (5.05%)  ← 표기가 “Land”로 들어온 케이스(정규화 여지)
8. Toyota (4.69%)
9. Lexus (4.58%)
10. Jeep (3.43%)

> 결론: 상위 브랜드 집중이 있지만, 전체적으로는 다수 브랜드가 섞인 **롱테일 카테고리 구조**입니다.  
> 따라서 CatBoost 또는 OneHot(min_frequency) 같은 전략이 유효합니다.

---

### 3.4 “설명 안 되는 저가” 룰 기반 점검
노트북에서는 “최근 연식인데 극단적으로 저렴한 price” 조건으로 의심 샘플을 탐색했지만,
- 결과: **suspicious rows = 0**

> 즉, 단순 규칙으로 잡히는 비정상 저가 샘플은 거의 없었고,  
> 오히려 고가/희귀 모델 등은 별도 전략(클리핑/가중치)로 다루는 편이 합리적일 수 있습니다.

---

## 4. Feature Engineering (설계 의도 + 구현)

### 4.1 차량 나이 & 주행 강도
- `age = CURRENT_YEAR - model_year`
  - 데이터 내 가장 최신 연도를 기준으로 age 정의
  - 음수 방지(`clip(lower=0)`)

- `log_milage = log1p(milage)`
  - 주행거리도 long-tail 형태 → 로그 변환으로 안정화

- `milage_per_year = milage / (age + 1)`
  - 같은 주행거리라도 연식에 따라 의미가 달라지므로 “연간 주행 강도”를 반영

- `log_milage_per_year = log1p(milage_per_year)`

---

### 4.2 엔진 문자열 파싱(engine parsing)
`engine` 텍스트에서 정규식을 이용해 다음을 추출:

- `engine_hp`: `(\d+\.?\d*)HP`
- `engine_liter`: `(\d+\.?\d*)L`
- `engine_cyl`: `(\d+)\s*Cylinder`

그리고 파생:

- `hp_per_liter = engine_hp / engine_liter`
- `hp_per_cyl = engine_hp / engine_cyl`

> 의도: 엔진은 가격을 강하게 설명하는 핵심 feature인데, 원본이 텍스트라 그대로 쓰기 어렵습니다.  
> 수치화하면 “출력/배기량/실린더”의 구조적 정보를 모델이 활용할 수 있습니다.

---

### 4.3 브랜드/모델 희소도(빈도 기반)
- `brand_freq = count(brand)`
- `model_freq = count(model)`

> 의도: 특정 브랜드/모델이 “자주 등장하는 대중 모델인지 / 희귀 모델인지”가 가격에 영향을 줄 수 있습니다.

---

### 4.4 사고/타이틀 이진화
- `has_accident`:
  - accident == `"At least 1 accident or damage reported"` → 1 else 0

- `is_clean_title`:
  - clean_title == `"Yes"` → 1 else 0

> 의도: 텍스트 카테고리를 “명확한 리스크 신호”로 압축해 모델이 쉽게 학습하도록 합니다.

---

### 4.5 색상(Black 여부) 플래그
- `is_black_ext`: 외장색에 “black” 포함 여부
- `is_black_int`: 내장색에 “black” 포함 여부

> 의도: 특정 인기 색상/인테리어 트림이 가격에 영향을 줄 수 있다는 가정.

---

## 5. FE 적용 전/후: 타깃과의 연관도 변화

노트북에서는 FE 전/후로 **`log(price)`와의 연관도(association)** 를 비교합니다.

FE 후 상위 연관 변수(노트북 출력 일부):

- `milage`: -0.695
- `age`: -0.664
- `model_year`: +0.664
- `log_milage`: -0.635
- `engine_hp`: +0.544
- `has_accident`: -0.296
- `engine_cyl`: +0.218
- `is_clean_title`: -0.201
- `engine_liter`: +0.172

> 해석: 연식/주행거리 뿐 아니라 **engine 파생 변수**들이 유의미한 신호로 떠오르며,  
> FE가 “가격을 설명하는 구조적 정보”를 추가해 준 것을 확인할 수 있습니다.

---

## 6. Preprocessing 전략

### 6.1 타깃 변환
- `y = log1p(price)`
- 예측 후 제출 시: `price_pred = expm1(pred_log)`

### 6.2 결측치 처리
- 범주형은 `"MISSING"`으로 통일해 “결측 자체를 하나의 카테고리”로 취급
- 수치형은 inf/-inf를 NaN 처리 후 imputer 적용

---

## 7. Modeling

### 7.1 LightGBM (Leak-free CV)

**핵심 포인트: 데이터 누수 방지**
- fold마다 `fit_transform / transform`을 분리해, 인코더/임퓨터가 validation 정보를 보지 않도록 구성합니다.

LightGBM 파라미터(노트북 FINAL CV CELL 기준):

- objective: regression
- metric: rmse
- boosting: gbdt
- learning_rate: 0.03
- num_leaves: 256
- min_data_in_leaf: 50
- feature_fraction: 0.8
- bagging_fraction: 0.8
- bagging_freq: 1
- lambda_l2: 1.0
- early_stopping_rounds: 300

✅ 결과 (5-Fold CV, RMSE on log1p):
- Fold별 RMSE: 0.486 ~ 0.490
- **CV RMSE(log1p): 0.4875877**

---

### 7.2 CatBoost (categorical/text 친화)

CatBoost는 범주형을 직접 다루는 데 강점이 있어,
- `cat_features`를 지정해 학습
- 일부 실험에서는 `engine`을 text feature로도 다루는 흐름이 포함되어 있습니다.

CatBoost 파라미터(노트북 CV 셀 기준):

- loss_function: RMSE
- iterations: 5000
- learning_rate: 0.05
- depth: 8
- l2_leaf_reg: 3
- early_stopping_rounds: 200
- verbose: False

✅ 결과 (5-Fold CV, RMSE on log1p):
- Fold1~5 RMSE: 0.4868 ~ 0.4894
- **CV mean: 0.4879865 (std: 0.0010641)**

---

## 8. Submission 생성(예측 파일)

노트북에서는 학습된 모델로 `test.csv`를 예측 후,
`sample_submission.csv` 포맷에 맞춰 `submission.csv`를 저장합니다.

- 예측은 `log1p` 스케일로 수행 → 마지막에 `expm1`로 원래 가격 스케일 복원
- 저장 파일: `submission.csv`

> ✅ 노트북 출력: “sample_submission.csv 포맷으로 submission 생성 / saved: submission.csv”

---

## 9. Repo에서 바로 재현하기 (How to Run)

### 9.1 Requirements
```bash
pip install -U pandas numpy scikit-learn lightgbm catboost matplotlib jupyter
```

### 9.2 폴더 구조(권장)
```
used_car/
  used_car.ipynb
  README.md
  data/
    train.csv
    test.csv
    sample_submission.csv
```

### 9.3 실행
```bash
jupyter notebook used_car.ipynb
```

> 노트북 일부 셀에는 개인 로컬 경로(`C:\Users\...`)가 존재합니다.  
> 레포지토리에서는 위 폴더 구조를 기준으로 **상대경로(`data/train.csv`)** 로만 읽도록 수정하는 것을 권장합니다.

---

## 10. 한계 & 개선 방향 (면접/발표 포인트)

### (1) Outlier 전략 비교
- 고가/저가 extreme은 제거 vs 클리핑 vs 가중치(loss reweight) 방식 비교
- 단, 단순 “최근 연식 저가” 룰로는 의심 데이터가 거의 없었음

### (2) 범주형 고카디널리티 처리 고도화
- `brand/model`에 Target Encoding(OOF 방식) + CatBoost/LGBM 비교
- 모델/트림 텍스트 정규화(예: “Land” → “Land Rover”)

### (3) 엔진 텍스트 파싱 보강
- “V6”, “I4”, “Turbo”, “Hybrid” 등 다양한 표기 대응
- 파싱 실패율(결측/NaN 비중) 점검 후 규칙 확대

### (4) 앙상블
- LGBM + CatBoost 평균/가중 평균
- 또는 OOF 기반 stacking

---

## 11. 참고
- 전체 실험/코드는 `used_car.ipynb`에 포함되어 있습니다.
