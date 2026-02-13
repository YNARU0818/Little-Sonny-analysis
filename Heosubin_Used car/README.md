# 🚗 중고차 가격 예측 모델 구축 프로젝트

이 프로젝트는 차량의 제원, 상태, 이력 데이터를 분석하여 중고차의 시장 가치(판매 가격)를 정확하게 예측하는 머신러닝 모델을 개발하는 것을 목표로 합니다.

---

### 1. 프로젝트 개요 (Project Overview)

* **목표**: 중고차 특성 데이터를 활용한 회귀(Regression) 모델링 및 가격 예측
* **데이터셋**: 중고차 사양 및 판매 가격 데이터 (Kaggle Playground Series 기반)
* **주요 도구**: Python, Pandas, Scikit-learn, XGBoost, CatBoost

---

### 2. 데이터 구성 및 특징 (Data Description)

데이터는 차량의 물리적 사양과 이력 정보를 포함하고 있습니다.

* **범주형 변수**: `brand`, `model`, `fuel_type`, `transmission`, `ext_col`, `int_col`, `accident`, `clean_title`
* **수치형 변수**: `model_year`, `milage`
* **타겟 변수**: `price` (연속형 수치)

---

### 3. 데이터 전처리 및 탐색 (EDA & Preprocessing)

데이터의 품질을 높이고 모델 학습에 적합한 형태로 가공하였습니다.

* **3.1 결측치 처리**: `fuel_type`, `accident` 등 결측값이 존재하는 컬럼을 `'Unknown'` 또는 적절한 상수로 대체
* **3.2 데이터 정제**: 제조사(`brand`) 명칭의 오기입 수정 및 텍스트 데이터 정규화
* **3.3 인코딩**: `LabelEncoder`를 사용하여 범주형 데이터를 수치 데이터로 변환
* **3.4 타겟 스케일링**: 가격 데이터의 편향(Skewness)을 해결하기 위해 `np.log1p` (로그 변환) 적용

---

### 4. 모델링 전략 (Modeling Strategy)

단일 모델의 한계를 극복하기 위해 다양한 알고리즘과 검증 기법을 도입했습니다.

* **4.1 사용 알고리즘**:
* **선형 모델**: Linear Regression, Ridge, Lasso
* **앙상블 모델**: RandomForest, GradientBoosting, **XGBoost**, **CatBoost**
* **기타**: SVR (Support Vector Regression), KNN


* **4.2 검증 기법**: **5-Fold Cross Validation**을 통해 모델의 일반화 성능 확보
* **4.3 최적화**: CatBoost 모델에 Early Stopping을 적용하여 과적합(Overfitting) 방지

---

### 5. 성능 평가 지표 (Evaluation Metrics)

회귀 분석의 정확도를 측정하기 위해 다음 지표를 사용했습니다.

* **MAE (Mean Absolute Error)**: 실제 가격과 예측 가격의 절대 오차 평균
* **RMSE (Root Mean Squared Error)**: 오차 제곱의 평균에 루트를 씌운 값
* **R² Score (결정계수)**: 모델이 데이터를 얼마나 잘 설명하는지 나타내는 지표 (최종 결과에서 **R² Score** 확인)

---

### 6. 프로젝트 결과 및 요약 (Results)

* 로그 변환된 예측값을 다시 `np.expm1`으로 복원하여 실제 가격 단위에서의 성능을 확인하였습니다.
* **CatBoost**와 **XGBoost** 모델이 가장 우수한 성능을 보였으며, 앙상블 기법을 통해 예측 안정성을 높였습니다.
* 데이터 전처리 과정에서의 결측치 처리와 브랜드 명칭 정제가 모델 성능 향상에 핵심적인 역할을 하였습니다.

---

### 7. 설치 및 실행 방법 (Usage)

```bash
# 1. 필수 라이브러리 설치
pip install pandas numpy matplotlib seaborn scikit-learn xgboost catboost

# 2. Jupyter Notebook 실행
jupyter notebook "중고차 가격예측 최종.ipynb"

```