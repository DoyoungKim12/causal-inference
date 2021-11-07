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
    -  <img src="https://github.com/DoyoungKim12/causal-inference/blob/master/img_BnT/bnt_1.PNG?raw=true">
    -  각 노드는 랜덤변수(random variable)이다. 화살표(또는 엣지)를 사용하여 특정 변수가 다른 변수의 원인이 되는지의 여부를 표시할 것이다. 
    -  좌측 예시에서는 Z가 X의 원인이고, U가 X와 Y의 원인이다.
    -  우측 예시는 방금 전의 예시를 설명한다. 중증도는 투약여부와 생존여부의, 투약여부는 생존여부의 원인이다. (아마 조건부 독립을 만족하지 않는 상태를 표현한 듯 하다.)
    -  이 인과관계 그래픽 모델 언어는 이처럼 인과관계에 대한 우리의 생각을 더 명확하게 하는데 도움을 줄 것이다.
<br><br><br><br>

## Crash Course in Graphical Models

- https://www.coursera.org/specializations/probabilistic-graphical-models
- 위 링크에 그래픽 모델의 자세한 강의가 있지만, 우리는 어쨌든 당장은 그래픽 모델이 어떤 종류의 독립성, 조건부 독립성 가정을 수반하는지를 이해하는 것이 중요하다. 독립성은 개울을 흐르는 물처럼 그래픽 모델을 통해 흐른다. 우리는 이 흐름을 멈출 수도 있고, 그 안에 있는 변수를 어떻게 다루느냐에 따라 활성화시킬 수도 있다. 
- 이를 이해하기 위해, 몇 가지 일반적인 그래프 구조와 예시를 살펴볼 것이다.  
<br><br>

- 먼저, 아래의 매우 간단한 그래프를 보자. A가 B의 원인이고, B가 C의 원인이 된다.
- <img src="https://github.com/DoyoungKim12/causal-inference/blob/master/img_BnT/bnt_2.PNG?raw=true">

  - 첫번째 그래프에서는 종속성(dependence)이 화살표의 방향대로 흐른다.<br> job promotion은 causal knowledge에 종속적이라고 말할 수 있다.
  - 자, 이제 중간 변수에 어떤 조건을 걸어보자(Y의 빨간색 표시). 이럴 경우, 위에서 말한 종속성이 차단된다. 따라서, X와 Z는 Y에 대해 독립이다. 위 그래프에서 빨간색은 Y가 조건부 변수임을 나타낸다.<br> 이처럼 solve problem이 조건부 변수가 된다면, (문제해결을 얼마나 잘하든간에) job promotion 여부가 causal knowledge의 수준이 어느 정도일지에 대한 정보를 주지 못한다. 
  - 일반화하면, 우리가 중간 변수 C를 조건화할때 A에서 B로 가는 직접 경로의 종속성 흐름은 차단된다.
    - <img src="https://render.githubusercontent.com/render/math?math=A \not\perp B, A \perp B|C">
<br><br>

- 이제 포크 구조(fork structure)를 보자. 이 경우에서는 하나의 변수가 두가지의 다른 변수의 원인이 된다. 여기서 종속성은 화살표 방향의 반대로 흐르고, 이를 백도어 경로(backdoor path)라고 부른다. 우리는 이 백도어 경로를 닫고 종속성을 차단할 수 있는데, 그 방법은 공통원인에 조건을 거는 것이다.
- <img src="https://github.com/DoyoungKim12/causal-inference/blob/master/img_BnT/bnt_3.PNG?raw=true">

- 예를 들어, 당신의 통계 지식이 인과추론과 ML을 더 잘 알게하는 원인이 된다고 하자. 내가 당신의 통계 지식 수준을 모른다고 할지라도, 당신이 인과추론을 잘 안다는 사실은 당신이 ML 또한 잘할 가능성이 높다는 것을 알려준다. 내가 당신의 통계 지식 수준을 모른다고 할지라도, 당신의 인과추론 지식의 수준을 통해 추론할 수 있는 것이다. (인과추론을 잘한다는 것은 아마 통계지식이 많다는 것을 의미할 것이고, 이는 곧 ML도 잘할 가능성이 있음)
- 이제 내가 이 통계지식에 조건을 걸면, ML을 잘 아는 것과 인과추론을 잘 아는 것은 독립인 상태가 된다. (이제 인과추론 지식 수준에 대해 아는 것은 ML 수준에 대한 어떤 힌트도 되지 못한다)
- 일반화하면, 같은 원인을 공유하는 두 변수는 종속적이지만, 그 원인을 조건화하면 독립이다.
  - <img src="https://render.githubusercontent.com/render/math?math=A \not\perp B, A \perp B|C">
<br><br>

- 이제 남은 것은 충돌 변수(collider)이다. 충돌은 두 개의 화살표가 하나의 변수에서 충돌하는 것을 의미한다. 우리는 이것을 두 변수가 공통의 효과를 공유한다고 말한다. 
- <img src="https://github.com/DoyoungKim12/causal-inference/blob/master/img_BnT/bnt_4.PNG?raw=true">

- 예를 들어, 승진을 하는 두가지 방법에 대해 생각해보자. 당신은 통계 지식이 높거나 상사에게 아첨할 수 있다. 승진에 조건을 걸지 않는다면, 당신의 통계 수준과 아첨 수준은 독립적이다. (통계 수준에 대해 안다고 해서, 아첨 수준에 대해 알 수 있는 것은 아니다.) 그러나, 만약 당신이 승진했고 당신의 통계 수준을 알게 된다면 당신의 아첨 수준을 알 수 있다. (통계를 못하는데 승진했다면 아첨을 잘했을 가능성이 크다) <br> 한 원인이 이미 효과를 설명하여 다른 원인이 발생할 가능성이 낮기 때문에, 이러한 현상은 가끔 explaining away라고 불린다. 
- 일반화하면, 충돌 변수를 조건화하는 것은 종속 경로를 여는 것이다. (조건을 걸지 않는 것은 그 경로를 닫힌 채로 두는 것이다.) 
  - <img src="https://render.githubusercontent.com/render/math?math=A \perp B, A \not\perp B|C">
<br><br>

- 이렇게 3개의 구조를 알면, 우리는 보다 일반화된 규칙을 도출할 수 있다. <br> 이제 경로는 다음과 같은 경우에만 차단된다.
  - 조건화된 non-collider를 포함한 경우
  - 조건화되지 않은 collider를 포함하고, 그 collider에 조건화된 자손 노드가 없는 경우
<br><br>

- 종속성이 그래프에서 어떻게 흐르는지를 보여주는 cheat sheet가 아래에 있다. <br> 파란 화살표의 끝에 선이 있는 경우는 독립을, 그렇지 않은 경우는 종속을 뜻한다. 
- <img src="https://github.com/DoyoungKim12/causal-inference/blob/master/img_BnT/bnt_5.PNG?raw=true">
<br><br>

- 마지막 예시로, 아래의 인과관계 그래프에서 종속과 독립관계를 알아맞혀보자.
- 문제
  - D와 C는 독립인가?
  - A 조건하에서 D와 C는 독립인가?
  - G 조건하에서 D와 C는 독립인가?
  - A와 F는 독립인가?
  - E 조건하에서 A와 F는 독립인가?
  - E,C 조건하에서 A와 F는 독립인가? 

- <img src="https://github.com/DoyoungKim12/causal-inference/blob/master/img_BnT/bnt_6.PNG?raw=true">

<br><br><br><br>

- 정답
  - 독립이다. 조건화되지 않은 collider를 포함하고 있다.
  - 독립이 아니다. 조건화된 collider를 포함하고 있다.
  - 독립이 아니다. 조건화된 노드가 collider의 자손 노드이다. (G를 A의 proxy로 보자)
  - 독립이다. B와 F가 독립이고, A와 B는 종속적이기 때문이다.
  - 독립이 아니다. 조건화된 E는 조건화된 collider이다.
  - 독립이다. E에 조건을 걸면 종속의 path가 열리지만, C에 조건을 걸면 그 path를 다시 닫게 된다. (B와 F가 종속이 되어도, A와 B가 독립이 되는 형태) 
<br><br>

- 인과 그래프 모델에 대해 알면, 인과추론에서 발생하는 문제를 이해할 수 있다. 우리가 본 바와 같이 문제는 항상 편향(bias)으로 귀결된다. 그래프 모델을 통해, 우리는 우리가 어떤 편향을 다루고 있는지, 그리고 이를 해결하기 위해 어떤 도구를 사용해야 하는지 알 수 있다. 
- <img src="https://github.com/DoyoungKim12/causal-inference/blob/master/img_BnT/bnt_7.PNG?raw=true">
<br><br>

