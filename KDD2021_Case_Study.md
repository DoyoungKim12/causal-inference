# Causal Inference and Machine Learning in Practice with EconML and CausalML
> Industrial Use Cases at Microsoft, TripAdvisor, Uber <br>
> https://causal-machine-learning.github.io/kdd2021-tutorial/

<br><br>

## Introduction to CausalML

<img src="https://github.com/DoyoungKim12/causal-inference/blob/master/img_BnT/kdd_1.PNG?raw=true" width=600><br><br>

<img src="https://github.com/DoyoungKim12/causal-inference/blob/master/img_BnT/kdd_2.PNG?raw=true" width=600><br><br>

- 데이터에 기반한 예측에서는 아래와 같은 질문에 답할 수 있다.
  - 다음 주에 받을 것으로 예상되는 주문량은?
  - 각 유저별로 예상되는 전환률은?

<br>

- 데이터에 기반한 **액션가능한 인사이트** 로는 아래와 같은 질문에 답할 수 있다.
  - 주문량을 늘리기 위해 우리가 무엇을 할 수 있을까?
  - 우리가 올림픽 광고를 좀 더 광범위하게 전개했다면 주문 건수는 얼마나 되었을까?
  - 특정 하위 모집단에만 프로모션 코드를 전송해도 (전체 유저에 전송하는 것과) 여전히 유사한 전환율 증가를 얻을 수 있을까?

<br><br>

<img src="https://github.com/DoyoungKim12/causal-inference/blob/master/img_BnT/kdd_3.PNG?raw=true" width=600><br><br>

- ATE 에서 CATE로 : 액션가능한 인사이트
  - 개인화된 경험을 가능하게 하며, 증분효과(액션을 하지 않았을 때와 대비했을 때, 액션으로 인해 얻게 된 이익)를 최적화함 

<br>

- 인과추론 + 머신러닝
  - 발견된 인사이트를 기반으로 한 최적화 어플리케이션 개발 

<br><br>

<img src="https://github.com/DoyoungKim12/causal-inference/blob/master/img_BnT/kdd_4.PNG?raw=true" width=600><br><br>

- 현실에서의 응용 사례
  - 개인화
    - (유저마다 다른) 처리효과를 학습하여 유저와 더욱 효과적으로 연결될 수 있음 
    - 예를 들어, 보다 의미있고 사전예방적인 고객 지원이 가능 
  - 유저 행동에 대한 관찰적 인과 추론
    - 유저 행동과 고객가치 분석 프로젝트에서, 교차 판매(한 제품을 구입한 고객이 다른 제품을 추가로 구입) 전환이 기존 유저의 장기적 가치에 미치는 영향을 측정
    - 첫번째 케이스 스터디 : CeViChE
  - 예산 최적화
    - 광고 타겟팅 : 보다 설득가능한 유저 그룹을 선택하여 광고 비용으로 얻는 리턴을 최적화 하기 위해 업리프트 모델링을 사용할 수 있음
    - 두번째 케이스 스터디 : Bidder
    - 프로모션 타겟팅 : 더 나은 처리 효과를 가질 것으로 추정된 유저에게 예산을 사용하기 위해 인과추론 추정을 사용함 

<br><br>

<img src="https://github.com/DoyoungKim12/causal-inference/blob/master/img_BnT/kdd_5.PNG?raw=true" width=600><br><br>

<img src="https://github.com/DoyoungKim12/causal-inference/blob/master/img_BnT/kdd_6.PNG?raw=true" width=600><br><br>

<img src="https://github.com/DoyoungKim12/causal-inference/blob/master/img_BnT/kdd_7.PNG?raw=true" width=600><br><br>

<img src="https://github.com/DoyoungKim12/causal-inference/blob/master/img_BnT/kdd_8.PNG?raw=true" width=600><br><br>

<br><br><br><br>

## Case Study #1: Causal Impact Analysis with Observational Data: CeViChE at Uber

<img src="https://github.com/DoyoungKim12/causal-inference/blob/master/img_BnT/kdd_9.PNG?raw=true" width=600><br><br>

<img src="https://github.com/DoyoungKim12/causal-inference/blob/master/img_BnT/kdd_10.PNG?raw=true" width=600><br><br>

- 실제 실험이 더 높은 내부 타당성을 가지지만, 실제로는 무작위로 처리를 제공하거나 보류하는 것이 불가능한 경우가 있음
- 관측적 인과 추론을 통해 인과 관계를 설정하고 효과(ATE)를 추정할 수 있음

<br><br>

<img src="https://github.com/DoyoungKim12/causal-inference/blob/master/img_BnT/kdd_11.PNG?raw=true" width=600><br><br>

- 교차판매의 기회는 크지만, 그 **장기적인 금전적 효과**를 엄격히 정량화할 것을 요구받음
  - 마케터 : 교차판매 예산기획 시점에서, 그들의 (신규)고객 획득을 위한 노력을 금전적 효과와 연결시킬 방법이 없음 (액션에 따라 어느 정도의 예산이 필요할지 모름)
  - 리더 : 제품 취득 간의 우선 순위를 매기기 위해 엄격한 인과 관계에 의존할 수 없음 (아무튼 우선순위를 정할 때 정량적 근거가 필요함)
  - 데이터 과학자 : 교차판매 전환 모델을 만드는 입장에서는, 개인 레벨의 전환이 가지는 개별 가치에 대한 인사이트 없이는 전환의 전체 기대값을 측정할 수 없음

<br><br>

<img src="https://github.com/DoyoungKim12/causal-inference/blob/master/img_BnT/kdd_12.PNG?raw=true" width=600><br><br>

- 통합 플랫폼으로의 우버에서 교차판매를 홍보하는 것이 비즈니스에 도움이 될지, 해가 될지를 알고 싶음
- 증분효과가 얼마나 될지를 알고싶음

<br><br>

<img src="https://github.com/DoyoungKim12/causal-inference/blob/master/img_BnT/kdd_13.PNG?raw=true" width=600><br><br>

- CausalML : 관찰 데이터를 활용하여 인과관계를 확인
  - 현장 실험과 연관된 시간축
  - 비용이 많이 들지 않음
  - (데이터와 실험설계에 따라 다르지만) 외적으로 타당한(검증된) 추론결과
  - 행동적 우려에 덜 시달림

<br><br>

<img src="https://github.com/DoyoungKim12/causal-inference/blob/master/img_BnT/kdd_14.PNG?raw=true" width=600><br><br>

- AB 테스트가 불가능할 때, 여러 시나리오에 적용되는 접근 방식의 일반화
  - Cross-sell/Upsell
  - Disengagement
  - Loyalty Program
  - Brand Impact
  - Marketing Campaigns

<br>

- 참고: 본 연구에서 분석한 데이터는 실험적이지 않은 관찰 데이터이다. 우리는 인과 관계를 추론하기 위해 최첨단 인과 추론 기술을 적용했다. 그러나, 현실이 추론 결과와 일치할 것이라 보장할 수 없으므로 수정된 전략을 평가할 때에는 A/B test를 권장한다.

<br><br>

<img src="https://github.com/DoyoungKim12/causal-inference/blob/master/img_BnT/kdd_15.PNG?raw=true" width=600><br><br>

- CeViChE 프레임워크의 구조

<br><br>

<img src="https://github.com/DoyoungKim12/causal-inference/blob/master/img_BnT/kdd_16.PNG?raw=true" width=600><br><br>

- Methodology (방법론)
  - CeViChE는 잘 구성된 튜닝 가능한 매칭기법과 엄격한 추론 검증기법을 사용함
    - Parametric Regression (옛날 방식 1)
    - Matching (옛날 방식 2)
    - 그리고 CeViChE
      - 계층화 PSM + Meta Learners + Sensitivity Analysis
      - (내가 생각하기에는 CausalML의 기능들을 단지 순서대로 사용했을 뿐인데, 패키지 내부 기능을 어떻게 엮어서 솔루션을 제공할지를 알려준다는 점에서는 의미가 있음)

<br><br>

<img src="https://github.com/DoyoungKim12/causal-inference/blob/master/img_BnT/kdd_17.PNG?raw=true" width=600><br><br>

- 첫번째 단계 : Analysis Setup
  - 피쳐를 모으고 정제하는 단계
  - 생각할 수 있는 모든 형태의 피쳐를 모음
  - 다른 관찰연구와 마찬가지로 X와 Y에 모두 영향을 주는 교란변수가 존재할지도 모르므로, 최대한 편향을 제거하기 위해 종합적인 정보를 수집함

<br><br>

<img src="https://github.com/DoyoungKim12/causal-inference/blob/master/img_BnT/kdd_18.PNG?raw=true" width=600><br><br>

- 두번째 + 세번째 단계 : PSM
  - Propensity score matching : 처리군과 대조군의 동질성을 보장하기 위한 유저 레벨의 매칭 과정
    - Propensity score는 각 유저의 과거 행동과 특성을 반영함
    - 해당 점수를 기반으로 최근접 이웃과 매칭 수행
    - 매칭 결과 최적화 (매칭 과정에 공변량을 추가)
    - 매칭 후 두 집단의 동질성을 각 피쳐의 분포로 확인 

<br><br>

<img src="https://github.com/DoyoungKim12/causal-inference/blob/master/img_BnT/kdd_19.PNG?raw=true" width=600><br><br>

- 행정 구역별로 동질한 처리군과 대조군을 생성

<br><br>

<img src="https://github.com/DoyoungKim12/causal-inference/blob/master/img_BnT/kdd_20.PNG?raw=true" width=600><br><br>

- 네번째 단계 : 추론
  - 다양한 방법을 활용하여 추론결과를 검증
  - 매칭 과정을 거친 데이터셋에 모델을 적용하여 ATE를 추정하고, 이 과정은 다양한 알고리즘과 시간 프레임에 의해 교차 검증됨
  - 메타 러너 알고리즘을 메인 추론 모델로 하여, 전환 이벤트가 예약에 미치는 영향을 추론함
  - 베이스 러너로 각각 XGBoost와 LR을 사용한 2개 모델 set으로 추론의 타당성을 평가함 (실제로 ATE를 측정할 일이 있을때 유용하게 참고할만한 구조)

<br><br>

<img src="https://github.com/DoyoungKim12/causal-inference/blob/master/img_BnT/kdd_21.PNG?raw=true" width=600><br><br>

- 다섯번째 단계 : 검증 
  - 추론의 강건성을 체크하기 위한 민감도 분석
    - Placebo Treatment
    - Replace/Add Irrelevant Confounder 
    - Subset Validation
    - Selection Bias

<br><br>

<img src="https://github.com/DoyoungKim12/causal-inference/blob/master/img_BnT/kdd_22.PNG?raw=true" width=600><br><br>

- Case Study
  - Monetary Impact of Rider to Eater (R2E) Cross Sell
  - 우버 사용자가 우버이츠를 사용하도록 권해서 교차판매가 일어날 경우 얻게되는 금전적 이익이 있는가?

<br>

- 분석 셋업
  - 처리 이전 기간 : 피쳐 수집 기간 (3개월)
  - 처리 기간 (3개월)
  - 처리 이후 기간 : GB 관찰
  - 처리군 : 처리기간에 rider에서 eater로 전환된 사람
  - 대조군 : 처리 이후 기간에도 rider에서 eater로 전환되지 않은 사람 

<br><br>

<img src="https://github.com/DoyoungKim12/causal-inference/blob/master/img_BnT/kdd_23.PNG?raw=true" width=600><br><br>

- PSM
  - 매칭은 처리군과 대조군 사이의 편향을 제거함
  - 매칭 이후 대조군의 수와 GB, Trip 수준이 동질하게 맞춰진 것을 확인

<br><br>

<img src="https://github.com/DoyoungKim12/causal-inference/blob/master/img_BnT/kdd_24.PNG?raw=true" width=600><br><br>

- Propensity Score의 분포를 통해 동질성 재확인

<br><br>

<img src="https://github.com/DoyoungKim12/causal-inference/blob/master/img_BnT/kdd_25.PNG?raw=true" width=600><br><br>

- 메타 러너
  - 다수의 메타 러너에서 같은 결과를 확인할 수 있음
  - 교차판매로 시너지 효과 발생 : 우버이츠로의 전환이 우버 사용으로 이어졌음 

<br><br>

<img src="https://github.com/DoyoungKim12/causal-inference/blob/master/img_BnT/kdd_26.PNG?raw=true" width=600><br><br>

- 민감도 분석
  - 민감도 분석에 의해 검증된 강건한 결과값

<br><br>

<img src="https://github.com/DoyoungKim12/causal-inference/blob/master/img_BnT/kdd_27.PNG?raw=true" width=600><br><br>

- 매칭결과 검증
  - 표준화된 평균차(SMD) 관찰

<br><br>

<img src="https://github.com/DoyoungKim12/causal-inference/blob/master/img_BnT/kdd_28.PNG?raw=true" width=600><br><br>

<br><br><br><br>

## Case Study #2: Targeting Optimization: Bidder at Uber

<img src="https://github.com/DoyoungKim12/causal-inference/blob/master/img_BnT/kdd_29.PNG?raw=true" width=600><br><br>

- 배경 : 광고주로서의 우버
  - 공급 측
    - 퍼블리셔 : NYT, WSJ, ESPN 등 광고가 노출되는 지면 또는 채널
    - Ad Exchange : AdX, MoPub 등 가장 높은 광고 단가를 제시한 광고주의 광고 게시
  - 수요 측
    - 우버 비딩 플랫폼
      - 유동적인 비딩 전략
      - ML 모델
      - 증분효과 측정

<br><br>

<img src="https://github.com/DoyoungKim12/causal-inference/blob/master/img_BnT/kdd_30.PNG?raw=true" width=600><br><br>

- 배경 : 미디어 바잉을 위한 우버 비더
  - 제한된 예산으로 누구에게 광고를 노출할지를 어떻게 결정할 것인가? 

<br><br>

<img src="https://github.com/DoyoungKim12/causal-inference/blob/master/img_BnT/kdd_31.PNG?raw=true" width=600><br><br>

- 문제 정의
  - 처리(광고노출) + 유저의 반응에 기반하여, 4개의 그룹으로 분류
    - Persuadable
    - Always-taker
    - Never-taker
    - Defier
  - 광고주로써, 이론적으로 우리는 Persuadable user에 관심이 있는데, 이들은 광고가 노출되었을 때 더 많은 이동/주문을 하는 유저들이다. 업리프트 모델은 이러한 유저들을  인과추론과 ML로 특정하기 위해 설계되었다.

<br><br> 

<img src="https://github.com/DoyoungKim12/causal-inference/blob/master/img_BnT/kdd_32.PNG?raw=true" width=600><br><br>

- Method : Overview
  - 실험 설정
    - 무엇을 모델링할 것인지
    - 모델링의 결과물을 어떻게 사용할 것인지 이해하기
    - 사업 지표 정의
    - 유저 정의
    - 처리(광고노출) 정의
  - 모델링
    - 정확도와 강건성 검증
    - 인사이트를 인과관계로 전환
  - 온라인 검증
    - 온라인 실험으로 업리프트 모델을 대조군과 비교하기

<br><br> 

<img src="https://github.com/DoyoungKim12/causal-inference/blob/master/img_BnT/kdd_33.PNG?raw=true" width=600><br><br>

- 실험 설정
  - 기간 구분 
    - 처리 전 기간 (4주)
    - 처리 기간 (4주)
    - 유동적인 처리 후 기간 (7일 또는 14일)
  - 실험군/대조군 구분
    - 실험군 : 처리 기간 내 최소 1회 이상 광고에 노출된 유저
    - 대조군 : 우리가 최소 1번이라도 광고 노출하려고 비딩했지만 실패한 유저

<br><br> 

<img src="https://github.com/DoyoungKim12/causal-inference/blob/master/img_BnT/kdd_34.PNG?raw=true" width=600><br><br>

- 데이터와 모델 설정
  - 피쳐 (gb는 gross booking의 약자)
    - gb_21d : 지난 21일 동안 사용자의 총 예약 수
    - order_21d : 지난 21일 동안 사용자의 총 우버이츠 주문 수
    - avg_waiting_time : 각 주문에 대한 사용자의 평균 대기 시간
    - avg_meal_subtotal : 각 식사에 대한 사용자의 평균 소계
    - total_ads_3m : 지난 3개월 동안 사용자에게 노출된 광고 수
    - city_avg_eater_gb : 과거 21일 동안의 도시별 평균 총 예약 수
  - 모델 변수
    - X : features
    - w : 처리 그룹
    - Y : 결과
    - p : 성향점수
    - tau : 처리효과 
 
<br><br> 

<img src="https://github.com/DoyoungKim12/causal-inference/blob/master/img_BnT/kdd_35.PNG?raw=true" width=600><br><br>

- 유저 타겟팅을 위한 CATE
  - CausalML은 각 메타러너가 CATE를 계산하기 위한 easy-to-use 인터페이스를 제공

<br><br>

<img src="https://github.com/DoyoungKim12/causal-inference/blob/master/img_BnT/kdd_36.PNG?raw=true" width=600><br><br>

- 검증 : 베이스 러너
  - 최종 CATE 예측결과 외에도, 베이스 러너의 예측결과는 러너 인스턴스를 통해 접근가능함 

<br><br>

<img src="https://github.com/DoyoungKim12/causal-inference/blob/master/img_BnT/kdd_37.PNG?raw=true" width=600><br><br>

- 검증 : Lift Curve
  - 리프트 곡선은 모집단을 최상의 리프트 성능에서 최악의 리프트 성능으로 정렬하고, 이를 각 세그먼트별로 분할하여 구축한다. y축은 각 세그먼트의 증분효과 이득을 나타낸다. 리프트 곡선은 직관적으로 이해하기 쉽지만, 해석하기는 어렵다.

<br><br>

- 이 외에도 다양한 검증 방법을 몇 개의 슬라이드를 더 할애하여 소개함
- 아무래도 본인들이 사용하는 방법론이 정말 믿을만 하다는 것을 설득하기 위해 여기에 많은 지면을 할애한 것으로 보임
  - Gain Chart
    - 이득(gain) 차트는 모집단을 최상에서 최악의 리프트 성능으로 정렬하고, 이를 세그먼트로 분할하여 보여준다. y축은 누적 증분 이득을 나타내고, x축은 타겟인 모집단의 비율을 나타낸다.
  - Qini Curve 
    - Qini-Coefficient는 Uplift 곡선과 랜덤 곡선의 아래 영역 넓이의 차이를 말한다.  Qini-Coefficient 값이 1에 가까울수록 업리프트 모델의 성능이 좋다는 뜻이다.
  - AUUC (Area Under Uplift Curve)
    - AUUC score와 Qini score는 두개의 각각 다른 업리프트 곡선의 아래 영역 넓이를 계산한다. 두 점수 모두 모델 성능을 비교하기 위한 정량 지표이다. 
  - TMLE for Robust Eval.
    - propensity score와 predicted outcome으로 doubly-robust한 ATE 추정치를 찾는 방법
    - TMLE Result : Gain Curve with TMLE as Ground Truth 

<br><br>

<img src="https://github.com/DoyoungKim12/causal-inference/blob/master/img_BnT/kdd_38.PNG?raw=true" width=600><br><br>

- 배포 : 타게팅 전략
  - 위 이미지에서 보는 차트가 Gain Curve with TMLE as Ground Truth 이다.
  - 상위 60%까지의 업리프트 스코를 가지는 타게팅 유저가 대부분의 ATE를 만들어낸다.
  - 이로써 40%의 예산을 아끼면서 광고효율을 67% 상승시킬 수 있다.

<br><br>

<img src="https://github.com/DoyoungKim12/causal-inference/blob/master/img_BnT/kdd_39.PNG?raw=true" width=600><br><br>

- 배포 : 온라인 검증
  - 상위 60%의 오디언스에만 광고를 노출했을 때의 ROAS가 2배 가량 높음 

<br><br>

<img src="https://github.com/DoyoungKim12/causal-inference/blob/master/img_BnT/kdd_40.PNG?raw=true" width=600><br><br>

- 배포 : 탐험/실행 설정
  - 각 페이즈별로 다른 모델을 적용
  - 전체의 80%에게는 모델이 찾아낸 높은 CATE가 기대되는 유저에게만 광고 노출 (exploit)
  - 나머지 20%는 정보를 모으기 위한 실험적 액션 적용 (explore)
    - 5%에는 모든 유저에 광고 노출
    - 5%에는 모든 유저에 비딩하지 않음
    - 10%에는 광고 노출 X 
    - 아무튼 이 정보를 활용하여 다음 페이즈의 모델을 생성

<br><br>
