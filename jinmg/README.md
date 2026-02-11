# 🚗 Used Car Price Prediction (Kaggle)

본 프로젝트는 캐글의 **Regression of Used Car Prices** 경진대회를 위해 진행되었으며, 중고차 시장의 다양한 지표를 분석하여 최적의 판매 가격을 예측하는 것을 목표로 합니다.

---

## 📊 1. 핵심 EDA 및 인사이트 (Finding the Money)
단순한 데이터 분포 확인을 넘어, 수익성과 직결되는 **'돈이 되는 지표'**를 발굴하는 데 집중했습니다.

* **감가상각 절벽(Depreciation Cliff):** 특정 연식(보증 종료 시점 등)에서 가격이 급격히 하락하는 구간을 포착.
* **사고 방어력(Resilience Analysis):** 브랜드별로 사고 이력이 중고차 가격에 미치는 영향도가 다름을 확인 (럭셔리 브랜드의 높은 가격 방어력).
* **주행 밀도(Mileage Intensity):** 단순 주행거리가 아닌, 연식 대비 주행거리를 분석하여 '저평가된 매물'의 특징을 정의.

---

## 🛠 2. 주요 피처 엔지니어링 (Feature Engineering)
모델의 성능을 극대화하기 위해 원본 데이터를 가공하여 고부가가치 변수를 생성했습니다.

1. **Mileage Intensity**: `milage / (2024 - model_year)` — 차량의 혹사 정도 수치화.
2. **Engine Specs Parsing**: 원본 텍스트 데이터(`engine`)에서 **Horsepower(HP)**, **Displacement(L)**, **Cylinders** 등 핵심 수치 추출.
3. **Log Transformation**: 가격 데이터의 심한 편향(Skewness)을 해결하기 위해 타겟 변수에 `log1p` 변환 적용.

---

## 🤖 3. 모델링 전략 (Modeling Strategy)
학습 효율성과 예측력 사이의 균형을 맞춘 **Hybrid 앙상블** 및 **교차 검증** 전략을 사용했습니다.

* **CatBoost Regressor**: 텍스트 중심의 범주형 데이터(`brand`, `model` 등) 처리에 최적화.
* **LightGBM**: 빠른 학습 속도와 수치형 데이터 변동성 포착을 위해 사용.
* **5-Fold Cross Validation**: 데이터의 모든 구간을 검증하여 과적합(Overfitting) 방지 및 리더보드 점수의 안정성 확보.



---

## 📈 4. 성능 평가 (Evaluation)
* **Main Metric**: RMSE (Root Mean Squared Error)
* **Secondary Metric**: $R^2$ Score (결정계수)
* **성과**: 로그 스케일 기준 $R^2$ 점수 **0.8X** 이상 달성 (검증 데이터셋 기준).

---

## 🚀 5. 실행 방법 (How to Run)
1. `train.csv`, `test.csv` 파일을 루트 폴더에 배치합니다.
2. 아래 패키지를 설치합니다:
   ```bash
   pip install catboost lightgbm pandas numpy scikit-learn