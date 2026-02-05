# ⚽ Project Little Sonny: 차세대 유망주 발굴 AI 솔루션

> **"데이터로 제2의 손흥민을 찾다"**
> 4,000여 명의 축구 선수 데이터를 분석하여 잠재력(Potential)을 예측하고, 최고의 공격수(FW) 유망주를 선별하는 머신러닝 프로젝트입니다.

---

## 📋 1. 프로젝트 개요 (Overview)
- **목적**: 선수들의 다양한 능력치 데이터를 기반으로 유망주(Prospect) 여부를 예측하고, 영입 우선순위를 산출합니다.
- **핵심 전략**: 
  - **도메인 피처 엔지니어링**: 나이 대비 효율성 및 공격 지능 지수 개발
  - **앙상블 모델링**: LGBM, XGBoost, CatBoost 조합을 통한 예측 안정성 확보
  - **AutoML 검증**: AutoGluon을 활용한 모델 성능 교차 검증

---

## 🚀 2. 분석 파이프라인 (Workflow)

본 프로젝트는 데이터 탐색부터 최종 리포트 도출까지 총 4단계의 프로세스로 구성됩니다.

1.  **데이터 탐색 (EDA)**: 포지션별(FW, MF, DF, GK) 스탯 분포 및 유망주 변별력 확인
2.  **특성 공학 (Feature Engineering)**: 선수의 잠재력을 나타내는 파생 변수 생성
3.  **모델 학습 (Modeling)**: 3종 앙상블 모델 및 AutoML 실험
4.  **최종 선정 (Scouting)**: 유망주 확률 기반 최상위 공격수 리스트 도출

---

## 📂 3. 파일 구성 및 실행 가이드 (File Structure)

가장 권장되는 실행 순서는 다음과 같습니다.

| 실행 순서 | 파일명 | 핵심 역할 |
|:---:|:---|:---|
| **Step 1** | `EDA_totalscore.ipynb` | **데이터 분석 및 전처리**: 포지션 그룹화 및 가중치 점수 설계 기초 작업 |
| **Step 2** | `sony (3).ipynb` | **메인 앙상블 모델링**: LGBM+XGB+CatBoost 조합 및 공격수 TOP 10 선정 |
| **Step 3** | `littlesonny3 (1).ipynb` | **AutoML 실험**: AutoGluon을 활용한 자동화 모델 학습 및 성능 비교 |
| **Step 4** | `Little-Sonny 전체 데이터 분석.ipynb` | **종합 리포트**: 전체 데이터 시각화 및 최종 Scouting Score 산출 |

---

## 💡 4. 핵심 분석 포인트 (Key Insights)

### 🧠 Feature Engineering
단순 스탯 합산이 아닌, 선수의 '가치'를 판단하기 위한 지표를 직접 설계했습니다.
- **`Age_Efficiency`**: (전체 능력치 / 나이) - 어린 나이에 높은 스탯을 보유한 선수를 포착
- **`Offensive_IQ`**: 위치 선정, 시야, 침착성 등 공격 전개에 필수적인 지능형 스탯 조합
- **`Scouting_Score`**: 모델이 예측한 확률과 주요 능력치를 결합한 최종 영입 점수

### 🤖 Modeling Strategy
- **Weighted Ensemble**: 예측 성능이 검증된 세 모델에 가중치(LGBM:35%, XGB:30%, CatBoost:35%)를 부여하여 오차를 최소화했습니다.
- **Position-Specific**: 공격수 포지션에 특화된 필터링을 거쳐 실질적인 영입 후보군을 압축했습니다.



📊 모델 성능 비교 및 분석 결과분석 
| 분석 단계 | 모델명 (Model) | 평가지표 (Metric) | 학습 점수 (Train) | 검증 점수 (Val/Test) | 주요 특징 |
|:---:|:---|:---:|:---:|:---:|:---|
| **Step 2** | **LightGBM** | F1-Score | 0.982 | 0.654 | 빠른 학습 속도, 기본 베이스라인 모델 |
| **Step 2** | **XGBoost** | F1-Score | 0.978 | 0.648 | 안정적인 성능, 정형 데이터 핵심 모델 |
| **Step 2** | **CatBoost** | F1-Score | 0.985 | 0.662 | 범주형 변수(Position) 처리에 탁월 |
| **Step 2** | **3-Model Ensemble** | F1-Score | **0.988** | **0.675** | 가중치(35:30:35) 결합, 예측 안정성 확보 |
| **Step 3** | **AutoGluon (L2)** | F1-Score | 0.998 | 0.684 | 스태킹(Stacking) 적용, 자동 피처 최적화 |
| **Step 3** | **AutoGluon (L3)** | F1-Score | **0.999** | **0.689** | **최종 최고 성능**, 다층 앙상블 전략 적용 |

💡 핵심 기술 포인트
다층 앙상블(Multi-layer Stacking): Step 2에서 개별 모델의 가중 합산(Ensemble)으로 기본 성능을 확보하고, Step 3에서 AutoGluon의 L3 스태킹을 통해 비선형적인 패턴까지 학습하여 최종 검증 점수를 끌어올렸습니다.

임계값 최적화(Threshold Tuning): 모델의 과적합(Train 0.999)을 경계하고, 실전 예측의 유연성을 위해 임계값을 0.38로 조정하여 유망주 발굴 누락을 최소화했습니다.

---
## 🏆 5. 최종 유망주 랭킹 (Sample)

분석 결과, 가장 높은 잠재력을 보인 공격수 TOP 5 리스트입니다.

| 순위 | ID | 나이 | 포지션 | 유망주 확률 | 특징 |
|:---:|:---:|:---:|:---:|:---:|:---|
| 🥇 | **TEST_1224** | 16 | ST | **98.9%** | 나이 대비 스탯 효율 압도적 1위 |
| 🥈 | **TEST_1066** | 16 | ST | 98.7% | 최상위권 Offensive IQ 보유 |
| 🥉 | **TEST_1548** | 17 | ST | 98.4% | 밸런스 잡힌 육각형 공격수 |
| 4 | **TEST_0930** | 17 | ST | 98.1% | - |
| 5 | **TEST_1594** | 18 | ST | 97.9% | - |

---

## 🛠 6. 기술 스택 (Tech Stack)

- **Language**: Python 3.x
- **Libraries**: 
  - `Pandas`, `NumPy` (Data Manipulation)
  - `Scikit-learn`, `LightGBM`, `XGBoost`, `CatBoost` (Machine Learning)
  - `AutoGluon` (AutoML)
  - `Matplotlib`, `Seaborn` (Visualization)

---
*본 프로젝트는 축구 데이터 분석을 통한 유망주 발굴을 목적으로 제작되었습니다.*