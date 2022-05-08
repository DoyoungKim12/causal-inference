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
