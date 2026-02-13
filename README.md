# 🚀 Little Sonny: Data Science & Machine Learning Project

**"도메인 지식 기반의 정밀 데이터 정제와 최적의 앙상블 모델링을 통한 인사이트 도출"**

본 프로젝트는 2주간 **축구 유망주 예측**, **학생 성적 데이터 검증**, **중고차 시세 예측**이라는 세 가지 데이터셋을 대상으로 데이터 사이언스 파이프라인 전반을 수행했습니다. 단순한 모델링을 넘어 데이터의 논리적 무결성을 검증하고, 현실 세계의 노이즈를 제어하여 모델의 신뢰성을 확보하는 데 집중했습니다.

### 📊 Project Summary at a Glance

| Project | Task | 핵심 기법 | 주요 성과 |
| --- | --- | --- | --- |
| **Soccer** | Classification | **AutoGluon(Stacking/Weighted Ensemble)**, Threshold 최적화 | F1-score 0.70+ 달성 및 유망주 랭킹 도출 |
| **Student** | Regression | 논리적 모순 역추적, Outlier Clipping | 데이터 무결성 확보 및 학습 패턴 인사이트 발굴 |
| **Used Car** | Regression | Regex 엔진 정보 파싱, Target Encoding(Median) | R2 Score 초기 모델 대비 성능 205% 향상 |

---

## ⚽ Project 1: Soccer Prospect Prediction (Classification)

FIFA 축구 선수 데이터를 분석하여 특정 선수가 향후 대성할 가능성이 있는 **'유망주(Prospect)'**인지 여부를 예측하고 최종 랭킹을 도출하는 프로젝트입니다.

### 🔍 탐색적 데이터 분석 (EDA) 및 변수 설계

* **나이(Age)의 지배적 영향력**: 16~18세 구간에서 유망주 비율이 압도적이며, 나이가 들수록 유망주 선정 비율이 급격히 하락하는 음의 상관관계를 확인.
* **Scouting Score 설계**: 단순 스탯 나열이 아닌 공격 결정력, 속도, 기술 등을 조합한 자체 지표를 생성하여 모델의 변별력을 높임.
* **포지션 그룹화(Pos_Group)**: 15개 이상의 세부 포지션을 Forward, Midfielder, Defender, GK로 단순화하여 학습 효율 최적화.

### 🤖 AutoGluon & AutoML 전략

* **AutoML 도입**: 여러 팀원의 협업 과정에서 **AutoGluon**을 활용하여 자동 전처리, 다중 모델 학습, 스태킹(Stacking) 앙상블을 수행.
* **WeightedEnsemble_L2**: 개별 모델의 장점을 결합한 가중치 앙상블을 통해 Validation F1 0.7273을 기록하며 최상위 성능 도출.
* **Feature Importance 해석**:
* **Age (0.322)**: 예측에 가장 결정적인 변수로 작용.
* **신체 조건**: Stamina(0.0166), Height(0.0135), Agility(0.0083)가 핵심 지표로 분석됨.



### 🛠 최종 모델링 및 랭킹 도출

* **가중치 앙상블**: AdaBoost(0.4)를 중심으로 RF(0.3), LGBM(0.3)을 결합하거나 AutoGluon의 스태킹 결과를 활용하여 Soft Voting 수행.
* **임계값 최적화**: Macro F1 점수를 극대화하기 위해 검증 데이터 기준 **최적 임계값 0.3434**를 적용하여 유망주 Top-N 랭킹 도출.

---

## 🎓 Project 2: Student Score Prediction (Regression)

Kaggle 데이터를 활용하여 학생들의 학습 습관과 생활 패턴이 **최종 시험 점수(exam_score)**에 미치는 영향을 분석하고 예측 모델을 구축했습니다.

### 📊 데이터셋 구성 (약 630,000개 레코드)

* **주요 변수**: `study_hours`, `class_attendance`, `sleep_hours`, `sleep_quality`, `internet_access`, `exam_difficulty` 등.

### 🔍 합성 데이터의 논리적 결함 및 인사이트

* **모순 데이터 발견**: 인터넷 미연결(`Internet: No`) 상태임에도 온라인 강의(`Online Video`)를 수강하는 데이터 8,730건 식별. 합성 데이터 생성 과정의 논리적 오류를 역추적함.
* **수면의 질 vs 시간**: 단순히 오래 자는 것보다 **수면의 질**이 높은 학생들의 성적이 일관되게 높게 나타남.
* **천재 그룹 분석**: 공부 시간이 매우 짧음에도(2시간 이하) 고득점인 그룹은 '코칭(Coaching)' 비율이 압도적으로 높음을 확인.

### 🛠 전처리 및 모델링 전략

* **이상치 Clipping**: 20점 이하 및 100점 이상 데이터를 삭제하지 않고 임계값으로 고정하여 합성 데이터의 분포 왜곡 방지.
* **사용 모델**: RandomForest, GradientBoosting, XGBoost 및 **LightAutoML(LAMA)**을 활용하여 대규모 데이터셋에 최적화된 회귀 성능 확보.

---

## 🚗 Project 3: Used Car Price Prediction (Regression)

18만 개의 대규모 중고차 데이터를 분석하여 가격을 예측하는 프로젝트로, 현실 세계의 노이즈를 제어하여 **MAE 22% 감소, R2 205% 향상**을 달성했습니다.

### 🛠 정밀 피처 엔지니어링 (Preprocessing Deep-Dive)

* **Regex 기반 엔진 정보 추출**: 비정형 텍스트인 `engine` 컬럼에서 정규표현식을 사용하여 핵심 수치 파싱.
* 마력(HP): `(\d+\.?\d*)HP` / 배기량(L): `(\d+\.?\d*)\s*(?:L|Liter)`


* **Target Encoding (Median)**: brand, model 변수는 이상치 영향을 최소화하기 위해 **중위값(Median)** 기준으로 매핑.
* **변속기 표준화**: 복잡한 변속기 타입을 Automatic, Manual, CVT, DCT, Other 5개 그룹으로 단순화.

### 🚫 도메인 기반 이상치 제거 (Outlier Removal)

* **가격 필터링**: $1,000,000 초과 매물 제거 및 일반 브랜드 중 $200,000 초과 매물(기재 오류) 제거.
* **사기/허위 매물 의심**: 럭셔리 브랜드임에도 $30,000 미만이거나, 연식은 오래되었으나 주행거리가 거의 없는 '좀비 카(Zombie Cars)' 데이터 필터링.

### 📈 모델링 전략 및 성과

* **사용 모델**: **CatBoost**(범주형 데이터 처리), **LightGBM**, **XGBoost** 가중치 블렌딩.
* **로그 변환**: 타겟 변수(price)에 `np.log1p()` 적용하여 오차 분포 정규화.
* **성과**: 5-Fold 교차 검증을 통해 로그 스케일 기준 검증 데이터셋 **R2 Score 0.8X 이상** 달성.

---

## 🎯 최종 결론 (Conclusion)

1. **데이터 품질의 중요성**: 모델 튜닝보다 선행되어야 할 것은 도메인 지식에 기반한 정교한 데이터 클리닝과 논리적 무결성 검증임을 확인했습니다.
2. **합성 데이터의 한계**: AI 시대의 분석가는 통계 수치뿐만 아니라 **현실적인 개연성**을 심판하는 역할을 수행해야 합니다.
3. **데이터 정제의 재정의**: 정제는 단순 수치 삭제가 아닌 비상식적 값을 걸러내어 데이터의 본질적 가치를 회복하는 과정입니다.

---

**Team Little Sonny** | 허수빈(팀장), 진민경, 최서연, 한승우

---

