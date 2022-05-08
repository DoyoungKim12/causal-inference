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
