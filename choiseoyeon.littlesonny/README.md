# 축구선수 유망주 예측 (Football Prospect Prediction)

## 프로젝트 개요
이 프로젝트는 축구 선수들의 다양한 신체 조건과 능력치 데이터를 분석하여 **유망주(Prospect)** 여부를 예측하는 머신러닝 프로젝트입니다.
`littlesonny3.ipynb` 노트북을 통해 데이터 전처리, 탐색적 데이터 분석(EDA), 모델링 과정을 진행하며, 최종적으로 테스트 데이터에 대한 예측 결과를 생성합니다.

특히, 본 분석에서는 **골키퍼(Goalkeeper)** 포지션에 주력하여 데이터를 심층 분석했습니다. 골키퍼의 고유 능력치와 유망주 여부 간의 관계를 파악하고, 이를 바탕으로 예측 성능을 높이기 위한 시도를 포함하고 있습니다.

## 파일 구성
- **littlesonny3.ipynb**: 메인 분석 및 모델링을 수행하는 Jupyter Notebook 파일
- **data/train.csv**: 모델 학습을 위한 훈련 데이터 (3019개)
- **data/test.csv**: 유망주 예측을 수행할 테스트 데이터 (1626개)

## 데이터셋 설명
데이터셋은 축구 선수의 기본 정보와 다양한 스탯 정보를 포함하고 있습니다.
- **기본 정보**: `ID`, `Age`, `Height`, `Weight`, `Position`, `PreferredFoot` 등
- **능력치**: `PaceTotal`, `ShootingTotal`, `PassingTotal`, `DribblingTotal`, `DefendingTotal`, `PhysicalTotal` 등
- **평점**: 포지션별 평점 (`CMRating`, `CBRating`, `GKRating`, `RMRating` 등)
- **Target**: `Prospect` (0: 유망주 아님, 1: 유망주)

## 주요 분석 내용
1. **데이터 로드 및 확인**:
   - Train/Test 데이터를 불러오고 데이터의 크기(Shape)와 형태를 확인합니다.
   - 결측치 및 기초 통계량을 확인합니다.

2. **탐색적 데이터 분석 (EDA)**

   데이터의 특성을 파악하고 모델링 전략을 수립하기 위해 다양한 각도에서 시각화를 진행했습니다.

   | 분석 대상 | 분석 내용 | 사용 기법/모델 | 시각화 (Visualization) |
   |:---:|:---|:---|:---|
   | **Target** | `Prospect` 클래스(0/1) 비율 확인 | 데이터 불균형 확인 | `sns.countplot` |
   | **GK 포지션** | **나이(`Age`) vs 능력치(`GKRating`)** | `Position == 'GK'` 필터링 분석 | `sns.scatterplot` (hue='Prospect') |
   | **GK 포지션** | **유망주 판별 핵심 변수 추출** | **Decision Tree Classifier** <br> (Feature Importance) | `sns.barplot` (Top 10 Features) |
   | **신체 정보** | `Age`, `Height`, `Weight` 분포 | 수치형 변수별 유망주/비유망주 비교 | `sns.boxplot`, `sns.histplot` |
   | **포지션** | 포지션별 주요 스탯 분포 비교 | 그룹별 통계 분석 | `sns.boxplot` |

   > **💡 골키퍼(GK) 심층 분석 주요 결과**:
   > - **피지컬의 중요성 발견**: 초기 예상(`GKRating` 중요)과 달리, 데이터 분석 결과 **피지컬 능력치(`Physical_Power`)**가 유망주 판별에 더 결정적인 요소임을 확인했습니다.
   > - **나이와 성장 가능성**: 어린 나이임에도 압도적인 피지컬을 보유한 선수가 유망주일 확률이 높았습니다.
   > - **파생 변수 생성 전략**:
   >   - **공통**: 각 포지션별 핵심 능력치를 종합한 점수와 나이 대비 효율성을 나타내는 지표를 생성했습니다.
   >   - **골키퍼(GK)**: `Physical_Power`, `BuildUp_Score`, `Physical_/_Age`, `Genius_Score`
   >   - **수비수(DF)**: `Iron_Wall_Genius` (수비 천재성), `Physical_Defending_Ratio` (피지컬+수비 효율), `Mobile_Defender_Score` (기동형 수비 점수)
   >   - **미드필더(MF)**: `Real_Genius` (미드필더 천재성), `Speed_Weight_Ratio` (피지컬 효율), `Stat_per_Age` (나이 대비 종합 능력)
   >   - **공격수(FW)**: `Shooting_Genius` (슈팅 천재성), `Aggression_Style` (공격 성향), `Tank_Index` (돌파력), `Stat_per_Year` (성장세)

3. **데이터 전처리 (Preprocessing)**:
   - 결측치 처리
   - 파생 변수 생성 (예: 능력치 대비 나이, 피지컬 점수 등)
   - 범주형 변수 인코딩 (Label Encoding / One-Hot Encoding)
   - 스케일링 (Scaling)

4. **모델링 (Modeling)**:
   - **예측 모델 학습** (AutoGluon 또는 Sklearn 기반 모델 활용)
   - **GK 포지션 모델 성능 평가 (Sklearn Baseline)**:
     
     | Model | Accuracy | F1-Score |
     |:---|:---:|:---:|
     | **AdaBoost** | **0.7778** | **0.6667** |
     | **RandomForest** | 0.7619 | 0.6809 |
     | **LogisticRegression** | 0.7460 | 0.6667 |
     | **GradientBoosting** | 0.7460 | 0.6364 |
     | **ExtraTrees** | 0.7302 | 0.6531 |
     | **LightGBM** | 0.7302 | 0.6531 |
     | **DecisionTree** | 0.6984 | 0.6122 |
     | **XGBoost** | 0.6667 | 0.5882 |


   - **AutoGluon 기반 포지션별 최적 모델 및 성능**:
    
     | Position | Best Model | F1-Score |
     |:---|:---|:---:|
     | **Attacker** | WeightedEnsemble_L2 | **0.8332** |
     | **Midfielder** | LightGBMXT_BAG_L2 | **0.7881** |
     | **Defender** | WeightedEnsemble_L2 | **0.8090** |
     | **GK** | WeightedEnsemble_L2 | **0.7797** |


5. **결과 예측 및 제출**:
   - 테스트 데이터에 대한 예측 수행
   - 유망주 확률 상위 선수 추출 및 결과 파일 저장

## 사용 라이브러리
- **데이터 처리**: pandas, numpy
- **시각화**: matplotlib, seaborn
- **머신러닝**: sklearn, autogluon (노트북 실행 환경에 따라 설치 필요)

## 실행 방법
1. 필수 라이브러리를 설치합니다.
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn autogluon
   ```
2. `littlesonny3.ipynb` 파일을 Jupyter Notebook 또는 JupyterLab에서 엽니다.
3. 셀을 처음부터 순서대로 실행하여 분석 및 모델링 과정을 수행합니다.
4. `data/` 폴더에 `train.csv`와 `test.csv` 파일이 위치해야 합니다.
