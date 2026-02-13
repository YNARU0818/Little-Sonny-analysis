# 🚀 Little Sonny: Data Science & ML Project

**"데이터 무결성 검증부터 고도화된 앙상블까지: 데이터의 가치를 복원하는 여정"**

---

### 📊 Project Overview

| 구분 | ⚽ 01. Soccer | 🎓 02. Student | 🚗 03. Used Car |
| --- | --- | --- | --- |
| **Task** | 유망주 분류 (Class) | 성적 예측 (Reg) | 시세 예측 (Reg) |
| **핵심 기법** | **AutoGluon** & Threshold | **Logic-Check** & Clipping | **Regex 파싱** & Encoding |
| **최종 성과** | **F1-score 0.70+** | **데이터 무결성 확보** | **R2 Score 0.79~0.80** |

---

## 1️⃣ [Classification] Soccer Prospect Prediction

> **"나이라는 제약 조건을 넘어 최적의 유망주 랭킹을 도출하다"**

### 🔍 Problem & Strategy

* **Problem**: 나이(Age) 변수의 지배력이 너무 커서 다른 신체 능력치가 무시되는 경향 발생.
* **Strategy**:
* **AutoGluon**을 활용한 자동화된 전처리 및 **Stacking/Weighted Ensemble** 구축.
* 단순 정확도보다 실제 유망주를 놓치지 않기 위한 **Recall 기반 Threshold(0.3434) 최적화**.



### 📈 핵심 분석 결과

* **Feature Importance**: Age(0.322) 외에도 Stamina, Height 등 신체 지표의 영향력 확인.
* **Final Impact**: 다수 모델의 장점을 결합하여 유망주 판별의 정밀도와 재현율을 동시에 확보.

---

## 2️⃣ [Insight] Student Performance Analysis

> **"합성 데이터 속 숨겨진 모순을 찾아 분석의 당위성을 입증하다"**

### 🔍 Problem & Strategy

* **Problem**: 약 63만 개의 방대한 합성 데이터에서 발생한 논리적 오류(인터넷X인데 온라인강의O 등) 발견.
* **Strategy**:
* **Logic-based Filtering**: 모순 데이터 8,730건을 역추적하여 합성 데이터의 특성 파악.
* **Clipping**: 극단적인 이상치를 삭제하지 않고 임계값으로 고정하여 데이터 분포의 왜곡 방지.



### 📈 핵심 분석 결과

* **인사이트**: 성적에 가장 큰 영향을 미치는 것은 '공부 시간'보다 **'수면의 질'**과 **'전문 코칭'**의 유무임이 밝혀짐.
* **Final Impact**: 단순 예측을 넘어 교육 환경 개선을 위한 실질적인 행동 지표 제시.

---

## 3️⃣ [Regression] Used Car Price Prediction

> **"비정형 노이즈를 정제하여 초기 모델 대비 성능을 205% 향상시키다"**

### 🔍 Problem & Strategy

* **Problem**: 18만 개의 대규모 데이터 내 비정형 텍스트와 허위 매물(이상치)로 인한 낮은 예측 성능.
* **Strategy**:
* **Regex 파싱**: `engine` 컬럼에서 마력(HP)과 배기량(L) 등 핵심 수치 변수 직접 추출.
* **Domain Filtering**: 브랜드 가치와 연식에 맞지 않는 '좀비 카' 및 허위 고가/저가 매물 대거 제거.
* **Target Encoding**: 브랜드/모델 변수를 중위값(Median)으로 매핑하여 이상치 저항성 확보.



### 📈 핵심 분석 결과

* **Model**: CatBoost, LightGBM, XGBoost 가중치 블렌딩 적용.
* **Final Impact**: 초기 R2 점수 0.26에서 **최종 0.79~0.80**으로 **약 205% 성능 향상** 달성.

---

## 🎯 최종 결론 (The Core Message)

* **Step 1. 도메인 기반 정제**: 모델 튜닝보다 선행되어야 할 것은 도메인 지식에 기반한 정교한 데이터 클리닝입니다.
* **Step 2. 데이터 무결성**: 분석가는 통계 수치 너머의 **현실적인 개연성**을 판단하여 데이터의 가치를 복원해야 합니다.
* **Step 3. 앙상블의 힘**: 단일 모델의 한계를 극복하기 위한 다각도의 앙상블(AutoGluon, Blending)이 최종 성능의 핵심입니다.

---

**Team Little Sonny** | 허수빈, 진민경, 최서연, 한승우

---
