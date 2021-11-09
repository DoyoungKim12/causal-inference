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
      - 이러한 증상별 조건화를 통해, 처리(효과를 판단하는) 메커니즘은 랜덤화만큼 좋아진다. (윤리적인 문제는 있겠지만,) 이제 중증 환자들은 그들이 중증이라서가 아니라 단순한 우연에 의해 투약 여부가 결정된다. 약을 투여받는 일과 (약을 받든 안받는 나오는)결과, 예를 들면 사망여부는 조건적으로 독립이라는 말이다. 약들 투여받은 그룹이 약을 투여받지 않았을 때의 사망률(반사실적 결과)과, 약을 투여받지 않은 그룹의 사망률이 어느 정도 비슷할 것이다. 
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
<br><br><br><br>

## Confounding Bias
<img src="https://github.com/DoyoungKim12/causal-inference/blob/master/img_BnT/bnt_8.png?raw=true">
(대충 둘다 쓰레기같아서 못고르겠다는 뜻)

<br><br>


- 편향의 첫번째 큰 원인은 교란(confounding)이다. 
- 교란은 처리와 결과가 같은 공통 원인을 공유할 때 발생한다.
- 예시 (교육수준과 수입의 관계)
  - 처리가 교육이고, 결과가 수입이라고 하자. 교육수준이 봉급에 미치는 영향을 알아내기는 힘든데, 그 이유는 그 둘이 같은 원인(지능 a.k.a. 능지)를 공유하기 때문이다. 따라서 우리는 더 교육받은 사람들이 더 돈을 잘 버는 이유가 단순히 그들이 더 똑똑해서이지, 교육수준이 높아서가 아니라고 주장할 수 있다. 인과효과를 판별하기 위해, 우리는 효과와 결과 사이의 모든 백도어 경로를 닫아야 한다. 우리가 그렇게 하기만 한다면, 남는 유일한 효과는 T -> Y의 직접효과 뿐이다. 우리의 예시에서 우리가 지능을 조건화한다면, 즉 같은 수준의 지능을 가졌지만 다른 수준의 교육을 받은 사람들을 비교한다면, 결과의 차이는 오직 교육에 의해서만 발생한다. 왜냐하면 지능은 모든 사람이 같은 수준이기 때문이다. 교란 편향을 수정하기 위해서는 이처럼 처리와 결과의 모든 공통 원인을 조건화해야 한다.
  - <img src="https://github.com/DoyoungKim12/causal-inference/blob/master/img_BnT/bnt_9.PNG?raw=true">
<br><br>

  - 불행하게도, 모든 공통변수를 처리하는게 항상 가능하진 않다. 가끔은 우리가 모르는 원인, 또는 알아도 측정할 수 없는 원인들도 존재한다. (지능의 경우는 후자이다.) 이제부터는 측정할 수 없는 변수를 U라고 쓸 것이다. 지능이 교육수준에 직접적인 영향을 주지 못한다고 해보자. 높은 지능은 SATs를 잘보게 해주고, SATs는 교육수준에 영향을 준다. 우리가 측정할 수 없는 지능을 처리하지 못한다고 하더라도, 우리는 SAT 수준을 통제할 수 있고 그를 통해 백도어 경로를 차단할 수 있다. 
  - <img src="https://github.com/DoyoungKim12/causal-inference/blob/master/img_BnT/bnt_10.PNG?raw=true">
<br><br>

  - 위 그래프에서 X1과 X2(각각 SAT와 가족 수입)을 조건화하는 것은 모든 백도어 경로를 닫기 에 충분하다. (즉, X1과 X2 조건 하에 결과와 처리는 독립이다) 따라서, 우리가 모든 공통 원인을 측정할 수 없더라도, 측정되지 않는 변수가 처리에 미치는 영향을 중재하는 측정가능한 변수를 통제하여 조건부 독립성에 다다를 수 있다. 
<br><br>
  
- 하지만 만약 측정불가능한 변수가 처리와 결과에 직접적인 영향을 미친다면 어떨까? 아래의 예시에서는 지능이 교육과 수입에 직접 영향을 미친다. 따라서 처리(교육)와 결과(봉급) 사이의 관계에 교란(confounding)이 발생한다. 이런 경우에는 교란 변수를 직접 통제할 수는 없지만, 우리에게는 교란변수의 대리 변수(proxy)처럼 기능하는 측정가능한 변수가 있다. 이러한 변수들은 백도어 경로에 있진 않지만, 이들을 통제하는 것이 편향을 감소시키는 데에 도움을 줄 수 있다. (편향을 아예 제거할 수는 없겠지만) 이러한 변수들은 가끔 대리 교란변수(surrogate confounder)라고 언급되기도 한다. 우리의 예시에서는 부모의 교육수준이 대리 변수가 될 것이다. 
- <img src="https://github.com/DoyoungKim12/causal-inference/blob/master/img_BnT/bnt_11.PNG?raw=true">
<br><br>

## Selection Bias
- 당신은 교란 편향이 없도록 당신의 모델에 측정가능한 모든 것들을 다 집어넣는 것이 좋은 아이디어라고 생각할 수 있다. 음, 다시 한번 생각해보자.
<img src="https://github.com/DoyoungKim12/causal-inference/blob/master/img_BnT/bnt_12.png?raw=true">
<br><br>

- 편향의 두번째 큰 원인은 선택편향(selection bias)이다. 
- 보통 우리가 해야 하는 것보다 더 많은 변수를 통제할 때 선택편향이 발생한다. 처리와 결과는 독립일 수 있지만, 우리가 충돌변수를 조건화하는 순간 종속적으로 변한다. 
- 예시 (교육수준과 임금, 그리고 투자)
  - 어떤 기적에 의해, 임금에 교육수준이 미치는 영향을 측정하기 위한 교육수준의 랜덤화가 가능해졌다고 하자. 하지만 혹시나 모를 교란을 피하기 위해, 수많은 변수들을 통제하기도 했다고 하자. 그 변수들 중에서 투자가 있었는데, 투자는 교육수준과 임금의 공통 원인이 아니다. 오히려 두 변수의 공통 결과에 가깝다. (더 많이 교육받은 사람은 더 많이 벌고 더 많이 투자한다. + 더 많이 버는 사람은 더 많이 투자한다.) 투자는 충돌변수이기 때문에, 이 변수를 조건화함으로써 처리와 결과 사이의 두번째 통로(종속성)를 열어버리게 된다. 그리고 이는 직접 효과를 측정하기 더 어렵게 만든다. 
  - 투자를 통제하여, 투자 수준이 같은 적은 수의 그룹에서 교육의 효과를 찾으려고 할 수 있다. 그러나 이렇게 하면, 당신은 간접적으로 (혹은 무심코) 임금의 변화를 줄여버리게 된다. (투자 수준이 비슷한 그룹은 임금도 비슷할 것) 따라서, 교육수준이 임금을 어떻게 변화시키는지 관찰할 수 없게 될 것이다. 임금 수준이 변화하는 것을 허락하지 않았기 때문이다. 
  - <img src="https://github.com/DoyoungKim12/causal-inference/blob/master/img_BnT/bnt_13.PNG?raw=true">
  <br><br>

  - 왜 이게 문제가 되는지를 입증하기 위해, 투자와 교육수준이 각각 2가지 경우만 있다고 가정하자. (사람들은 투자를 하거나 안하거나, 교육을 받거나 안받거나 둘중 하나이다) 처음에는 우리가 투자를 통제하지 않으므로, 교육수준이 랜덤화되었다는 가정 아래 편향 term(<img src="https://render.githubusercontent.com/render/math?math=E[Y_0|T=1] - E[Y_0|T=0] = 0">)은 0이다. 이것이 의미하는 바는 그들이 교육을 받든 받지 않았든간에, 그들이 교육을 받지 않았을 때의 임금은 같다는 것이다. 하지만 투자를 통제한다면 어떤 일이 발생할까?
  <br><br>

  - 아마 아래의 수식과 같이 교육을 받지 않은 쪽의 임금이 더 높은 것처럼 관찰되는 상황이 벌어질 것이다.
  - <img src="https://render.githubusercontent.com/render/math?math=E[Y_0|T=0,I=1] > E[Y_0|T=1,I=1] = 0">
  - 말로 풀어서 설명하자면, 투자하는 사람들 중에서 교육 없이도 그렇게까지 할 수 있는 사람들은 더 많은 벌이를 얻기 위해 교육과는 별개의 삶을 산다. 이러한 이유로, 교육받지 않은 사람들의 임금은 교육받은 사람들이 교육을 받지 않았을 때를 가정한 임금보다 높을 것이다.   
<br><br>

  - 그래픽 모델을 사용해서 설명하자면, 투자를 조건화함으로써 높은 교육수준이 낮은 임금과 연관성을 갖게 되고 우리는 음의 편향을 관찰하게 된다. (곁들여 말하자면, 공통 효과의 어느 자손 노드에 조건화를 하더라도 같은 결과가 나오게 된다.)
  - <img src="https://github.com/DoyoungKim12/causal-inference/blob/master/img_BnT/bnt_14.PNG?raw=true">
<br><br>

- 비슷한 현상은 처리의 매개 변수(mediator)를 통제할 때에도 나타난다. 매개변수는 처리와 결과의 흐름 사이에 끼어있는 변수이다. 
- 예시 (교육수준과 임금, 그리고 사무직)
  - 이번에도 역시 교육수준의 랜덤화가 가능했다고 하자. 그러나, 역시 혹시 모를 교란에 대비해 사무직 여부도 통제하기로 했다고 하자. 이러한 조건 설정 역시 인과효과 측정을 편향되게 만든다. 이 경우에는 충돌변수로 프론트도어 경로를 열었기 때문이 아니라 처리가 작동하는 채널 하나를 닫았기 때문이다. 이 예시에서 사무직은 더 높은 교육수준이 더 높은 임금을 받도록 이끄는 길 중 하나이다. 이것을 통제함으로써, 우리는 그 채널을 닫고 오직 교육이 임금에 미치는 직접적 영향만을 남겨두게 된다. 
  - <img src="https://github.com/DoyoungKim12/causal-inference/blob/master/img_BnT/bnt_15.PNG?raw=true">
<br><br>

  - 잠재적 결과 논거를 제시하기 위해, 우리는 랜덤화로 인해 편향이 0이라는 사실을 안다고 하자. 그러나 우리가 사무직으로 개인을 조건화(통제)한다면, 아래의 수식과 같이 교육을 받지 않은 그룹의 임금이 더 높은 것처럼 관찰될 것이다.
  - <img src="https://render.githubusercontent.com/render/math?math=E[Y_0|T=0,WC=1] > E[Y_0|T=1,WC=1] = 0">
  - 왜냐하면 교육수준이 낮음에도 사무직 일자리를 쟁취한 사람들은 아마도 같은 일자리를 위해 교육을 받아야했던 사람들보다 더 열심히 일해왔을 것이기 때문이다. (그리고 같은 일자리를 위해 교육을 받아야했던 그 사람들이 교육을 안받았다면? 당연히 연봉은 더 낮지 않았을까)
<br><br>

- 위 경우에서, 매개변수를 통제하는 것은 음의 편향을 발생시킨다. 이는 교육의 효과가 실제보다 더 작아보이게 하는 효과를 가져온다. 그리고 이는 교육의 인과효과과 양의 방향이기 때문이다. (음의 방향이었다면 양의 편향을 가져왔을 것) 모든 경우에서, 이러한 종류의 조건화는 효과를 원래보다 더 약해보이게 만든다.
<br><br>

(마무리는 편향 개노답 삼형제... 왼쪽부터 차례대로 충돌변수 조건화로 인한 선택편향, 교란편향, 매개변수 조건화로 인한 선택편향)
<img src="https://github.com/DoyoungKim12/causal-inference/blob/master/img_BnT/bnt_16.png?raw=true">
<br><br><br><br>

## Key Ideas
- 인과관계 아이디어를 더 잘 이해하고 표현하기 위한 언어로써 그래프 모델을 공부해봤다.
- 우리는 그래프 상에서의 조건부 독립의 규칙에 대해 간단히 요약했다.
- 이 규칙들은 우리를 편향으로 이끄는 3가지 구조를 탐구하는 데에 도움이 되었다.
  - 첫번째는 교란으로, 처리와 결과가 같은 원인을 공유할 때 발생한다.
  - 두번째는 공통효과를 조건화하여 발생하는 선택편향이다.
  - 세번째는 매개변수에 대한 과도한 통제로 발생하는 선택편향이다.
<br><br>

## References + Contribute
- 본문을 참고(https://matheusfacure.github.io/python-causality-handbook/04-Graphical-Causal-Models.html)







