⚽ 차세대 공격수 유망주 발굴 프로젝트 (LAMA 기반)
본 프로젝트는 축구 선수의 다양한 스탯 데이터를 분석하여 **잠재력이 높은 공격수 유망주(Prospect)**를 예측하는 머신러닝 파이프라인입니다. LightAutoML(LAMA) 프레임워크를 활용하여 예측 모델의 정확도를 극대화했습니다.

🚀 프로젝트 개요
단순히 종합 능력치(STRating)만 보는 것이 아니라, 나이 대비 기술적 완성도와 심리적 요인을 결합한 자체 지표를 생성하고, 최신 AutoML 기법을 적용해 실제 유망주가 될 확률을 산출합니다.

🛠 주요 특징 (Key Features)
*고급 피처 엔지니어링: 단순 스탯을 넘어선 4가지 핵심 도메인 지표 생성
    *Finishing_Power: 결정력과 슛 파워의 조화
    *Offensive_IQ: 위치 선정, 시야, 반응 속도를 통한 공격 지능
    *Physical_Potential: 신체적 완성도 및 에너지 레벨
    *On_the_Ball: 드리블 및 볼 컨트롤 능력

*Prospect_Index: 나이 가산점을 포함한 자체 개발 잠재력 지수 적용
*LightAutoML(LAMA) 활용: LightGBM과 CatBoost의 스태킹(Stacking) 모델을 통한 고성능 이진 분류

📊 분석 프로세스
1. Data Loading: train.csv 및 test.csv 로드
2. Feature Engineering: 공격 생산성, 신체 조건, 기술적 잠재력 등 파생 변수 생성
3. AutoML Training: 5분간(timeout=300) 최적의 모델 탐색 및 5-Fold 교차 검증
4. Target Prediction: 테스트셋 대상 유망주 확률(Prospect_Prob) 예측
5. Filtering: 핵심 공격수 포지션(ST, LW, RW, CF, LF, RF) 집중 분석

📈 모델 성능
*평가 지표: ROC-AUC Score
*본 모델은 검증 데이터셋(OOF)에서 높은 AUC를 기록하며 공격수의 미래 가치를 안정적으로 평가합니다.

📁 출력 결과물
*fw_prospect_top10.csv: 모델이 선정한 차세대 공격수 유망주 TOP 10 명단
    *포함 데이터: 선수 ID, 나이, 포지션, 유망주 확률, 잠재력 지수 등