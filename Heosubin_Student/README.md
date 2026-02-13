# 🎓 학생 시험 점수 예측 프로젝트 (Student Score Prediction)

본 프로젝트는 학생들의 인구통계학적 정보, 학습 습관, 생활 패턴 및 시험 환경 데이터를 활용하여 **최종 시험 점수(exam_score)를 예측**하는 머신러닝 모델을 구축하는 것을 목표로 합니다.

## 1. 프로젝트 개요

* **목적**: 학생의 학습 및 생활 데이터를 기반으로 성적 예측 모델 개발
* **데이터셋**: Kaggle Playground Series - Season 6, Episode 1
* **주요 과업**: 데이터 전처리, 탐색적 데이터 분석(EDA), 다양한 머신러닝 알고리즘 비교 및 최적화

## 2. 데이터셋 구성 (Features)

데이터셋은 약 630,000개의 레코드로 구성되어 있으며, 주요 컬럼은 다음과 같습니다.

1. **학생 기본 정보**: `age`(나이), `gender`(성별), `course`(수강 과정 - B.Tech, B.Sc 등)
2. **학습 패턴**: `study_hours`(공부 시간), `class_attendance`(출석률), `study_method`(학습 방법), `internet_access`(인터넷 여부)
3. **생활 습관**: `sleep_hours`(수면 시간), `sleep_quality`(수면의 질), `facility_rating`(시설 만족도)
4. **시험 환경 및 결과**: `exam_difficulty`(시험 난이도), **`exam_score` (Target - 최종 점수)**

## 3. 기술 스택 및 라이브러리

* **Language**: Python
* **Data Analysis**: Pandas, NumPy
* **Visualization**: Matplotlib, Seaborn
* **Machine Learning**:
* Scikit-learn (RandomForest, GradientBoosting, LogisticRegression, SVM, KNN)
* XGBoost
* LightAutoML (LAMA)


* **Environment**: Jupyter Notebook, Kaggle API

## 4. 수행 단계 (Workflow)

1. **데이터 로드 및 확인**: Kaggle API를 통한 데이터 다운로드 및 결측치 확인 (결측치 없음 확인)
2. **파생 변수 생성 (Feature Engineering)**:
* `attendance_bin`: 출석률을 구간별로 범주화
* `sleep_range`: 수면 시간을 'Short', 'Good' 등으로 범주화


3. **데이터 전처리**:
* 범주형 데이터 인코딩 (Label Encoding, One-Hot Encoding 등)
* 수치형 데이터 스케일링 (StandardScaler)


4. **모델 학습 및 평가**:
* 다양한 분류/회귀 모델 학습
* VotingClassifier를 이용한 앙상블 기법 적용


5. **최종 예측 및 제출**: `test.csv` 데이터를 활용한 예측 수행 및 `submission.csv` 저장

## 5. 주요 코드 예시

```python
# 파생 변수 생성 예시
test_df['attendance_bin'] = pd.cut(test_df['class_attendance'], bins=range(40, 110, 10)).astype(str)
test_df['sleep_range'] = pd.cut(test_df['sleep_hours'], bins=[0, 4, 7, 10, 24], labels=['Very Short', 'Short', 'Good', 'Long']).astype(str)

```

## 6. 결과 및 성과

* 테스트 데이터셋에 대한 컬럼 매칭 및 예측 완료
* `LightAutoML` 등을 활용하여 모델 성능 최적화 시도

---