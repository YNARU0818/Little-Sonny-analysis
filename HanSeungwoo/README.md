# ⚽ Soccer Prospect Prediction Project

본 프로젝트는 **축구 선수 유망주(Prospect) 예측**을 목표로 한 데이터 사이언스 프로젝트입니다.  
EDA 기반 점수 설계부터 머신러닝/AutoML 모델링, 최종 유망주 랭킹 도출까지  
**End-to-End 분석 파이프라인**을 구현했습니다.

---

## 📌 프로젝트 개요

- **목표**
  - 선수 능력치 데이터를 활용해 *유망주(Prospect)* 여부 예측
  - 확률 기반 **유망주 랭킹(Top-N)** 생성

- **핵심 아이디어**
  1. 포지션별로 중요한 능력치는 다르다
  2. 모델 이전에 **EDA 기반 점수(TotalScore_EDA)** 를 설계
  3. 해석 가능한 지표 + 머신러닝 + AutoML 결합

---

## 🔍 프로젝트 흐름

1. **EDA & Feature Understanding**
2. **EDA 기반 점수 설계 (TotalScore_EDA)**
3. **전통 ML 모델 비교 (Cross-Validation)**
4. **AutoGluon 기반 앙상블**
5. **Feature Importance 해석**
6. **유망주 Top-N 랭킹 도출**

---

## 📊 1️⃣ 모델별 교차검증 성능 비교

여러 전통 머신러닝 모델을 대상으로  
**Stratified Cross-Validation**을 수행하여 성능을 비교했습니다.

### 📈 교차검증 성능 (mean ± std)

| Model | Accuracy | F1-score | ROC-AUC | Precision | Recall |
|---|---|---|---|---|---|
| RandomForest | 0.7748 ± 0.0035 | 0.6667 ± 0.0121 | 0.8387 | 0.7162 | 0.6248 |
| ExtraTrees | 0.7751 ± 0.0123 | 0.6691 ± 0.0234 | 0.8415 | 0.7137 | 0.6312 |
| GradientBoosting | 0.7804 ± 0.0120 | 0.6798 ± 0.0162 | 0.8453 | 0.7190 | 0.6459 |
| **AdaBoost** | **0.7854 ± 0.0108** | **0.6997 ± 0.0132** | 0.8451 | 0.7077 | **0.6927** |
| LightGBM | 0.7705 ± 0.0107 | 0.6651 ± 0.0115 | 0.8322 | 0.7040 | 0.6312 |
| XGBoost | 0.7668 ± 0.0103 | 0.6602 ± 0.0106 | 0.8198 | 0.6984 | 0.6275 |
| LogisticRegression | 0.7807 ± 0.0100 | 0.6745 ± 0.0139 | **0.8497** | 0.7276 | 0.6294 |
| SVM | 0.7814 ± 0.0129 | 0.6603 ± 0.0211 | 0.8406 | **0.7528** | 0.5890 |

### 🔎 해석
- **AdaBoost**
  - F1-score / Recall 기준 최고 성능
  - 유망주 누락(False Negative)을 줄이는 데 유리
- **Logistic Regression**
  - ROC-AUC 최고 → 전반적 분리 능력 우수
  - 해석 가능한 베이스라인 모델
- **Tree 계열 모델**
  - 비선형 관계 학습에 강점
  - 안정적인 성능 분포 확인

---

## 🤖 2️⃣ AutoGluon (AutoML) 결과

AutoGluon을 활용해  
**자동 전처리 + 다중 모델 + 스태킹 앙상블**을 수행했습니다.

### 📊 AutoGluon 모델 성능 (F1 기준)

| Model | Holdout F1 | Validation F1 | Stack Level |
|---|---|---|---|
| LightGBMLarge_BAG_L2 | **0.7079** | 0.7109 | 2 |
| RandomForestEntr_BAG_L2 | 0.7045 | 0.7058 | 2 |
| NeuralNetTorch_r30_BAG_L1 | 0.7016 | 0.7205 | 1 |
| **WeightedEnsemble_L2** | 0.7005 | **0.7273** | 2 |
| NeuralNetFastAI_BAG_L2 | 0.6989 | 0.7146 | 2 |

### 🔎 해석
- **WeightedEnsemble_L2**
  - Validation 기준 F1 최고
  - 개별 모델의 장점을 결합한 안정적인 성능
- 단일 모델 대비 **일관된 성능 향상** 확인
- 실전 적용에 가장 적합한 구조

---

## 🧠 3️⃣ Feature Importance 분석

AutoGluon 기준,  
예측에 **가장 큰 영향을 미친 변수들**은 다음과 같습니다.

| Feature | Importance | p-value |
|---|---|---|
| **Age** | **0.322** | **< 1e-6** |
| Stamina | 0.0166 | 0.0055 |
| Height | 0.0135 | 0.0054 |
| Agility | 0.0083 | 0.0031 |
| PosGroup | 0.0078 | 0.0164 |
| Balance | 0.0067 | 0.0068 |
| Weight | 0.0056 | **< 1e-4** |
| Position | 0.0052 | 0.058 |
| TotalScore_EDA | 0.0049 | 0.288 |
| Finishing | 0.0047 | 0.056 |

### 🔎 해석
- **Age가 압도적으로 가장 중요한 변수**
- 신체 조건(Height, Weight, Balance) + 활동량(Stamina)
- 포지션 정보(PosGroup, Position)가 보조적 역할
- **EDA 기반 TotalScore_EDA는 단독보단 보조 신호로 작용**

---

## 🏆 4️⃣ 유망주 Top-10 랭킹 (Train 기준)

Prospect 확률 기준 상위 10명 유망주입니다.

| Rank | ID | Position | PosGroup | Age | Prospect_Prob |
|---|---|---|---|---|---|
| 1 | TRAIN_2840 | CAM | MF | 16 | 0.8897 |
| 2 | TRAIN_1255 | CAM | MF | 16 | 0.8883 |
| 3 | TRAIN_1231 | CAM | MF | 16 | 0.8837 |
| 4 | TRAIN_1521 | CAM | MF | 16 | 0.8804 |
| 5 | TRAIN_0330 | CAM | MF | 16 | 0.8788 |
| 6 | TRAIN_3009 | RM | MF | 16 | 0.8774 |
| 7 | TRAIN_0047 | CAM | MF | 16 | 0.8756 |
| 8 | TRAIN_0420 | RM | MF | 16 | 0.8727 |
| 9 | TRAIN_0381 | ST | FW | 16 | 0.8704 |
| 10 | TRAIN_1506 | CAM | MF | 16 | 0.8666 |

### 🔎 관찰 포인트
- **Top-10 중 9명이 MF**
- 모두 16세 → *Age 효과 명확*

---
