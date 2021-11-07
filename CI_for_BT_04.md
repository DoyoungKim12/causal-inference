<br><br><br><br>

# 04 - Graphical Causal Models

<br><br>

## Thinking About Causality
- 그래픽 모델(Graphical models)은 인과관계의 언어이다. 
  - 일단 시작으로, 잠재적 결과(potential outcomes)의 조건부 독립성에 대해 생각해보자. 이는 인과추론을 할 때 참이 되어야 하는 주요 가정 중 하나이다.
  -  <img src="https://render.githubusercontent.com/render/math?math=(Y_0,Y_1) \perp T|X">
    - 조건부 독립은 처리(treatment) 자체로 인한 효과만을 측정할 수 있게 해준다. 
    - 예시 (약의 효과 측정하기)
      - (많이 아픈 사람에게만 약을 준다고 가정하는 듯 하다.) 중증 환자만 약을 복용한다면, 약을 주는 것이 마치 환자의 건강을 해치는 것처럼 보인다. 병의 중증도 효과와 약의 효과가 뒤섞이기 때문이다. (경증그룹은 약 안먹고 멀쩡, 중증그룹은 약 먹고 사망)
      - 그러나 환자를 경증과 중증으로 분류하여 각 그룹의 투약 효과를 분석하면 실제 효과를 보다 명확하게 파악할 수 있다.
      - 여기서 환자의 상태에 따라 모집단을 세분화하는 것을 X에 대한 제어, 또는 조건화라고 한다. (controlling for or conditioning on X)
      - 이러한 증상별 조건화를 통해, 처리(효과를 판단하는) 메커니즘은 랜덤화만큼 좋아진다. (윤리적인 문제는 있겠지만,) 이제 중증 환자들은 그들이 중증이라서가 아니라 단순한 우연에 의해 투약 여부가 결정된다. 약을 투여받는 일과 (약을 받든 안받는 나오는)결과, 예를 들면 사망여부는 조건적으로 독립이라는 말이다. 
<br><br>

- 독립성과 조건부 독립성은 인과추론의 핵심이지만, 이 추상적인 개념들을 이해하기는 쉽지 않다. 하지만 우리가 이 문제를 묘사하는 올바른 언어를 사용한다면 상황이 달라질 수 있다. 여기서 인과관계 그래픽 모델이 등장한다. 인과 그래픽 모델은 무엇이 무엇의 원인인지의 관점에서 어떻게 인과관계가 작동하는지를 표현하는 방법이다.
  - 그래픽 모델은 아래의 그림과 같다.
    -  <img src="https://github.com/DoyoungKim12/causal-inference/blob/master/img_BnT/bnt_1.png"





- Zeus의 가족을 우리의 관심 집단으로 설정하자.
    - <img src="https://github.com/DoyoungKim12/causal-inference/blob/master/image_CH1/capture_0.PNG?raw=true">

    - 처리를 시행했을 때와 그렇지 않을 때를 이처럼 모두 알고 있다고 가정하자.

    - 위 표에서 <img src="https://render.githubusercontent.com/render/math?math=Pr[Y^{a=1}=1] = 0.5">임을 확인할 수 있다.
        - 즉, 처리(심장이식)를 시행했을 때, 위 사람들 중 반절은 죽는다.
    - 마찬가지로, <img src="https://render.githubusercontent.com/render/math?math=Pr[Y^{a=0}=1] = 0.5">라는 것도 확인할 수 있다.
        - 즉, 처리(심장이식)를 시행하지 않았을 때에도, 위 사람들 중 반절은 죽는다.

    - 이제 우리는 평균 인과효과를 정의할 수 있다.
        - 결과 Y에 대한 처리 A의 **평균 인과효과** : <img src="https://render.githubusercontent.com/render/math?math=E[Y^{a=1}] \neq E[Y^{a=0}]">
        - 따라서, 위 예시의 처리 A는 평균 인과효과가 없다고 할 수 있다.
