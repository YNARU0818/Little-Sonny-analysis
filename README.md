# 🚗 중고차 가격 예측 (Used Car Price Prediction)

이 저장소는 `car_prices.ipynb` 노트북을 통해 중고차 가격을 예측하는 머신러닝 프로젝트를 담고 있습니다.  
XGBoost, LightGBM, RandomForest 등 다양한 회귀 모델을 실험하고, 최종적으로 앙상블(Ensemble) 기법과 AutoGluon(AutoML)을 적용하여 예측 성능을 극대화하는 과정을 상세히 기술합니다.

---

## 📋 1. 프로젝트 개요 (Overview)

- **목표**: 차량의 다양한 정보(브랜드, 모델, 연식, 주행거리 등)를 기반으로 중고차 가격(`price`)을 예측합니다.
- **데이터셋**: [Kaggle](https://www.kaggle.com/) 또는 유사한 출처의 중고차 데이터셋을 사용합니다.
- **해결 문제**: 회귀(Regression)
- **평가 지표**: **RMSLE** (Root Mean Squared Log Error), **R² Score**
  - 가격 데이터의 특성상 값이 크고 분포가 넓으므로, 로그 변환된 오차인 RMSLE를 주 지표로 사용합니다.

---

## 📊 2. 데이터셋 명세 (Dataset Description)

데이터셋은 총 **188,533개**의 학습 데이터와 **125,690개**의 테스트 데이터로 구성되어 있습니다.

| 컬럼명 (Column) | 타입 | 설명 | 비고 |
| :--- | :--- | :--- | :--- |
| `id` | int | 고유 식별자 | 학습 시 제외 |
| `brand` | object | 제조사 브랜드 | 예: Ford, BMW, Mercedes-Benz |
| `model` | object | 모델명 | 예: F-150, 3 Series |
| `model_year` | int | 차량 연식 | 예: 2018, 2022 |
| `milage` | int | 주행 거리 (마일) | 가격과 강한 음의 상관관계 |
| `fuel_type` | object | 연료 종류 | Gasoline, Diesel, Hybrid 등 (결측치 존재) |
| `engine` | object | 엔진 스펙 | 예: 3.5L V6... (마력, 배기량 추출 필요) |
| `transmission` | object | 변속기 | A/T, M/T 등 |
| `ext_col` | object | 외장 색상 | Black, White 등 |
| `int_col` | object | 내장 색상 | Black, Gray 등 |
| `accident` | object | 사고 이력 | 무사고 여부 (결측치 존재) |
| `clean_title` | object | 클린 타이틀 여부 | 소유권 상태 (결측치 다수) |
| **`price`** | int | **Target (가격)** | 예측해야 할 값 ($) |

---

## 🛠 3. 데이터 전처리 (Preprocessing)

성능 향상을 위해 다음과 같은 전처리 과정을 거쳤습니다.

### 3.1. 결측치 및 이상치 처리 (Handling Missing & Outliers)
### 3.1. 결측치 및 이상치 처리 (Handling Missing & Outliers)

| Feature | Issue | Handling Strategy |
| :--- | :--- | :--- |
| **`fuel_type`** | 결측치 존재 | 최빈값(Mode)인 'Gasoline'으로 채우거나, 'Unknown'으로 대체 |
| **`accident`** | 결측치 존재 | 문맥상 '무사고(None reported)'로 간주하여 처리 |
| **`clean_title`** | 80% 이상 결측 | 'Yes' 외에는 'No' 또는 'Unknown'으로 간주하여 이진 변수화 |
| **`clean_title`** | 80% 이상 결측 | 'Yes' 외에는 'No' 또는 'Unknown'으로 간주하여 이진 변수화 |

#### 3.1.2. 이상치 제거 심층 로직 (Outlier Removal Mechanics)
단순한 IQR 방식이 아닌, **도메인 특성을 반영한 정교한 필터링**을 적용하여 모델의 안정성을 높였습니다.

1.  **초고가 데이터 제거 (High Price Outliers)**
    - **통계적 접근**: 상위 **0.1%** (`quantile(0.999)`)에 해당하는 데이터는 일반적인 중고차 시장의 범위를 벗어난다고 판단하여 제거했습니다.
    - **가격 제한 (Hard Limit)**: **$1,000,000 (100만 불)** 이상의 매물은 분석 대상에서 제외했습니다.
    - **브랜드 기반 필터링**:
        - **가짜 슈퍼카 (Fake Supercars)**: 가격이 **$700,000**를 넘지만, 브랜드가 슈퍼카(Ferrari, Lamborghini 등)가 아닌 경우(예: 일반 브랜드의 기재 오류) 제거했습니다.
        - **일반 브랜드의 이상 고가**: 대중적인 브랜드(`common_brands`)임에도 가격이 **$200,000**를 초과하는 경우(튜닝카 또는 기재 오류) 제거했습니다.

2.  **초저가 및 의심 데이터 제거 (Low Price / Suspicious Data)**
    - **폐차 수준 (Junk Cars)**: 가격이 **$4,000** 미만인 차량은 정상적인 주행이 불가능하거나 부품용일 가능성이 높아 제외했습니다.
    - **럭셔리 사기 의심 (Luxury Scam)**: 럭셔리 브랜드(`luxury_brands`) 차량임에도 가격이 **$30,000** 미만인 경우(사고차, 침수차 또는 허위매물 가능성) 제거했습니다.
    - **신차급 저가 매물**: 주행거리가 짧거나 연식이 최신임에도 비정상적으로 가격이 낮은 데이터를 필터링했습니다 (e.g., `price < $5,000` & `milage < 100,000`).

3.  **데이터 모순 (Logical Errors)**
    - **Zombie Cars**: 연식이 오래되었는데(`model_year < 2010`) 주행거리가 거의 없는(`milage < 100`) 데이터는 입력 오류로 간주하여 제거했습니다.

### 3.2. 파생 변수 생성 (Feature Engineering)

단순한 수치 변환을 넘어, 도메인 지식을 활용한 파생 변수를 생성했습니다.

| Feature | New Variable | Logic / Formula |
| :--- | :--- | :--- |
| **`milage`** | `milage_log` | 왜도(Skewness) 완화를 위해 $log(x+1)$ 변환 적용 |
| **`brand`, `model`** | `brand_price_mean`, `model_price_mean` | 각 브랜드/모델별 평균 가격(Target Encoding)을 매핑하여 수치형 변수로 변환 |
| **`fuel_type`** | `fuel_type_ratio` | 전체 데이터 중 해당 연료 타입의 비율을 계산하여 희소 범주(Rare Category) 문제를 완화 |
| **`ext_col`, `int_col`** | `color_risk_score` | (가상의 예시) 인기 없는 색상 조합에 패널티 부여 또는 단순화(Simplication) |
| **`car_age`** | `model_year` | `2024 - model_year`로 차량 연식 계산 (감가상각 핵심 요인) |
| **`hp` (Horsepower)** | `engine` | 엔진 텍스트에서 마력(HP) 추출 (누락 시 평균값 대체) |
| **`displacement`** | `engine` | 엔진 배기량(L) 추출 |
| **`cylinder`** | `engine` | 실린더 개수(V6, V8 등) 추출 |
| **`transmission_g`** | `transmission` | 다양한 변속기 타입을 주요 카테고리(Auto, Manual 등)로 그룹화 |
| **Stats Features** | `brand`, `model` | 브랜드/모델별 평균 가격, 평균 주행거리 등을 새로운 피처로 추가 |

### 3.3. 핵심 전처리 기술 심층 분석 (Deep Dive into Key Techniques)

이 프로젝트에서는 단순히 라이브러리를 사용하는 것을 넘어, **도메인 지식에 기반한 정밀한 전처리**를 수행했습니다.

#### 1. 정규표현식(Regex)을 활용한 엔진 정보 추출
비정형 텍스트인 `engine` 컬럼에서 핵심 정보를 추출하기 위해 다음과 같은 정규표현식을 사용했습니다.
- **마력 (Horsepower)**: `r'(\d+\.?\d*)HP'` 패턴을 사용하여 숫자 데이터를 정밀하게 추출하였습니다.
- **배기량 (Liters)**: `r'(\d+\.?\d*)\s*(?:L|Liter)'` 패턴으로 'L' 또는 'Liter' 단위 앞의 수치를 파싱했습니다.
- **실린더 (Cylinders)**: 
  - `r'(\d+)\s*Cylinder'` (표준 표기)
  - `r'(?:V|I|H|W)(\d+)'` (V6, V8 등 엔진 형식 표기)
  - 위 두 가지 패턴을 모두 고려하여 결측 없는 데이터를 생성했습니다.

#### 2. 고차원 범주형 변수의 Target Encoding (Median)
`brand`(약 40개)와 `model`(수백 개 이상)과 같은 고차원(High-Cardinality) 변수는 원-핫 인코딩 시 차원의 저주에 빠질 위험이 있습니다. 이를 방지하기 위해 **Target Encoding**을 적용했습니다.
- **Why Median?**: 평균(Mean) 대신 **중위값(Median)**을 사용하여, 수억 원을 호가하는 슈퍼카와 같은 **가격 이상치(Outlier)의 영향력을 최소화**했습니다.
- **Code Logic**:
  ```python
  # Brand Encoding 예시
  brand_map = train.groupby('brand')['price'].median()
  train['brand_encoded'] = train['brand'].map(brand_map)
  ```

#### 3. 변속기(Transmission) 데이터 표준화
다양한 표기법으로 분산된 변속기 정보를 4가지 핵심 그룹으로 표준화하여 모델이 학습하기 쉽게 만들었습니다.
- **Mapping Logic**:
  - `CVT` $\rightarrow$ **'CVT'**
  - `Manual`, `M/T` $\rightarrow$ **'Manual'**
  - `DCT`, `Dual Shift` $\rightarrow$ **'DCT'**
  - `Automatic`, `A/T` $\rightarrow$ **'Automatic'**
  - 그 외 $\rightarrow$ 'Other'

### 3.4. 데이터 스케일링 및 변환 (Scaling & Transformation)

| Method | Target | Description & Benefit |
| :--- | :--- | :--- |
| **Log Transformation** | `price` (Target) | `np.log1p()` 적용. 오차 분포 정규화 및 RMSLE 성능 향상 |
| **Log Transformation** | `milage` | 왜도(Skewness)가 높은 주행거리 변수의 분포 완화 |
| **Standard/MinMax** | `numeric` feature | 선형 모델(Ridge, Lasso)의 수렴 속도 및 성능 최적화 |

---

## 🧪 4. 모델링 및 실험 (Modeling)

다양한 회귀 모델을 학습시키고 `K-Fold` 교차 검증(5-Fold)을 통해 성능을 비교했습니다.

### 4.1. 사용된 모델 (Models Used)
1. **XGBoost**: 강력한 성능의 부스팅 모델.
2. **LightGBM**: 빠르고 효율적인 부스팅 모델 (이번 프로젝트의 **Best Model**).
3. **RandomForest**: 안정적인 배깅(Bagging) 모델.
4. **ExtraTrees**: 무작위성을 더한 트리 앙상블.
5. **Linear Models**: Ridge, Lasso, LinearRegression (기준점 역할).

### 4.2. 성능 평가 (Evaluation)
- 각 모델별 RMSLE와 R² Score를 비교한 결과, **LightGBM**과 **XGBoost**가 가장 우수한 성능을 보였습니다.
- 시각화를 통해 모델별 에러율을 비교 분석했습니다.

### 4.3. 앙상블 (Ensemble)
- **가중치 블렌딩 (Weighted Blending)**: 단일 모델보다 더 강건한 예측을 위해, 1등 모델(예: LightGBM/XGBoost)에 60%, 나머지 상위 모델들의 평균에 40% 가중치를 주어 최종 예측값을 생성했습니다.
- **전체 재학습**: 검증이 끝난 후, 전체 Train 데이터를 사용하여 모델을 다시 학습시켜 실전 예측력을 높였습니다.

### 4.4. AutoGluon 적용
- 추가적으로 **AutoGluon** (Automated Machine Learning) 라이브러리를 적용하여, 자동으로 최적의 모델을 찾고 앙상블하는 코드를 포함했습니다.
- `presets='best_quality'` 설정을 통해 시간 내 최고 성능을 목표로 학습합니다.

---

## 🚀 5. 실행 방법 (How to Run)

이 프로젝트는 Jupyter Notebook 환경에서 실행되도록 작성되었습니다.

### 5.1. 환경 설정
필요한 라이브러리를 설치합니다.
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm autogluon
```

### 5.2. 노트북 실행
`car_prices.ipynb` 파일을 열고 셀을 순차적으로 실행합니다.
1. 데이터 로드 및 확인
2. 전처리 및 시각화 (EDA)
3. 모델 정의 및 교차 검증
4. 최종 앙상블 및 제출 파일(`fixed_submission.csv`) 생성
5. (선택) AutoGluon 실행 및 결과 확인

---

## 📈 6. 결과 요약 (Results)

- **Best Single Model**: LightGBM (RMSLE 약 0.44 대)
- **Ensemble Strategy**: Weighted Average (Main Model 0.6 + Others 0.4)
- **Files Generated**: 
  - `fixed_submission.csv`: 앙상블 모델의 예측 결과
  - `ensemble_result.csv`: 상위 5개 예측 샘플 확인용

---

> **Note**: 이 분석은 제공된 데이터셋에 최적화되어 있으며, 실제 중고차 시장의 모든 변수를 반영하지는 않을 수 있습니다. 데이터의 정제 방식과 모델의 하이퍼파라미터 튜닝에 따라 성능은 달라질 수 있습니다.
