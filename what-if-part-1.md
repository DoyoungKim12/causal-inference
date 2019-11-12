<br><br><br><br><br><br>

# What If - Part 1

<br><br><br><br>

# Part 1 : Causal inference without models

<br>

## Chapter 1. A DEFINITION OF CAUSAL EFFECT

### 인과효과의 정의

- 인과효과(관계)를 우리는 이미 체득하여 알고 있다.
    - 관찰을 통해 얻은 직관을 선택에 반영해왔으니까…!
- 이 장은 개념에 대한 깊은 통찰을 제공하기 위한 것이 아니다.
    - 이미 가지고 있는 직관을 **공식화**하는 수학적 표기법을 배울 것이다.
    - 책 전체에서 사용될 causal concept을 이해하려면 이 표기법들을 이해하는 것이 매우 중요하다.

<br>

### 1.1. Individual causal effects

### 개별적 인과효과

- Zeus는 심장이식이 필요한 환자이다. 1월 1일에 그는 새 심장을 이식받았고, 5일 후 사망하였다.
    - 우리가 일종의 신성한 계시를 통해, Zeus가 심장이식을 안받았더라면 5일 후에도 살아있었을 것을 알았다고 치자.
    - 이걸 알고 있다고 한다면, 대부분은 심장이식이 Zeus의 사망원인이라는 데에 동의할 것이다.
    - 심장이식 이라는 개입(Intervention)이 Zeus의 5일 간 생존에 인과효과를 가져왔다.

- 다른 환자, Hera가 있다. 그녀 또한 역시 1월 1일에 심장이식을 받았다. 5일 후, 그녀는 살아남았다.
    - 그런데 우리가 어떻게든, Hera가 1월 1일에 심장이식을 받지 않았어도 5일 후 생존했을 것이라는 사실을 알았다고 치자.
    - 따라서 심장이식은 헤라의 5일 간 생존에 인과효과를 가지지 않는다.

- 위 두 개의 일화는 사람들이 어떻게 인과효과를 추리(판단)하는지 보여준다.
    - 우리는 보통 action A가 일어났을 때의 결과와 일어나지 않았을 때의 결과를 비교한다.
        - 만약 두 결과에 차이가 있다면, 우리는 action A가 인과효과를 가진다, 또는 원인이 된다고 말한다.
        - 그렇지 않다면, 우리는 action A가 결과에 영향을 주지 못한다고 말한다.
    - 많은 사회과학자들은 action A를 **개입, 노출, 처리**라고 말한다. **(an intervention, an exposure, or a treatment)**

- 우리의 인과적 직관을 수리통계학적으로 이해할 수 있도록, 몇가지 표기법을 소개한다.
    - Binary Treatment A와 Binary Outcome Y를 가정하자.
    - 이 책에서는 A와 Y처럼 다른 개체에 대해 다른 값을 가질 수 있는 변수들을 **확률변수(random variable)**라 칭하겠다.
    - <img src="https://render.githubusercontent.com/render/math?math=Y^{a=1}"> : Treatment A가 1일 때(실행되었을 때)의 Y, <img src="https://render.githubusercontent.com/render/math?math=Y^{a=0}">도 이처럼 읽으면 된다.
        - 위 표기도 모두 확률변수
        - Zeus의 경우,  <img src="https://render.githubusercontent.com/render/math?math=Y^{a=1} = 1 \quad and \quad Y^{a=0} = 0">
        - Hera의 경우,  <img src="https://render.githubusercontent.com/render/math?math=Y^{a=1} = 0 \quad and \quad Y^{a=0} = 0">

- 이제 우리는 각 **개체(Individual)** 에 대한 **인과효과**를 정의할 수 있다.
    - Treatment A는  <img src="https://render.githubusercontent.com/render/math?math=Y^{a=1} \neq Y^{a=0}">이라면 개체의 Outcome Y에 대해 인과효과를 가진다.
    - <img src="https://render.githubusercontent.com/render/math?math=Y^{a=1}">과 <img src="https://render.githubusercontent.com/render/math?math=Y^{a=0}">은 각각 potential outcomes(잠재적 결과), 또는 counterfactual outcomes(반사실적 결과)라 칭한다.
        - 둘다 맞는 표현인데, 사람마다 선호하는 표현이 다를 뿐이다.

- 각 개체에게, 두 반사실적 결과 중 하나는 실제로 겪은 사실이다.
    - 예를 들어, Zeus는 실제로 심장이식을 받았기 때문에 <img src="https://render.githubusercontent.com/render/math?math=Y^{a=1}">과 Y가 1로 같다.
    - 이와 같은 같음(Equality)은 일치성(Consistency)으로 정의된다.
        - Consistency : If <img src="https://render.githubusercontent.com/render/math?math=A_{i}"> = a,  then  <img src="https://render.githubusercontent.com/render/math?math=Y^{a}_i = Y^{A_i} = Y_i">`

- 개별적 인과효과는 반사실적 결과들 2개를 대조하여 정의되지만, 각 개인에 대해서는 여러 결과들 중 하나만 관찰된다.
    - 다른 모든 반실현적 결과는 여전히 관찰되지 않는다.
    - **일반적으로는 개별적 인과관계를 파악할 수 없다**는 것이 결론이다.
        - 즉, 데이터가 누락되어 관측된 데이터의 함수로 표현될 수 없다. (가능한 예외는 Fine Point 2.1 참조)

<br>

### 1.2. Average causal effects

### 평균 인과효과

- 개별적 인과효과를 정의하기 위해서는 3가지 요소가 필요하다.
    - Outcome, Actions, Individual
    - 그러나, 개별적 인과효과를 파악하는 것이 일반적으로 불가능하기에 종합적 인과효과(aggregated causal effect)에 관심을 가져보기로 하자.
        - aggregated causal effect : 개체 집단 내의 평균 인과효과
        - 종합적 인과효과를 정의하기 위해서도 역시 3가지 요소가 필요하다
            - Outcome, Actions까진 같음
            - **Individual → well-defined population of individuals** (Action에 따른 Outcome 비교가 가능한!)

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

- 처리에 대한 결과가 여러 개일 경우, 2개로 정확히 정의하는 것이 필요하다.
    - 예를 들어, 아스피린을 (1정씩 5일 간격으로 3달간) 복용한 사람 vs. 복용하지 않은 사람 처럼 구체적으로 정의하는 것이 좋다.

- 평균 인과효과가 없다고 하여 개별 인과효과가 존재하지 않는 것은 아니다.
    - 만약 평균 인과효과가 없을 때 개별 인과효과까지 없다면, sharp causal null hypothesis가 성립한다고 표현한다.
    - 이제부터는 **평균 인과효과를 간단히 인과효과라 칭하겠다.**

<br>

### 1.3 Measures of causal effect

### 인과효과의 측정

- 위 사례에서 인과귀무가설(causal null hypothesis)이 기각되지 않은 이유를 다시 떠올려보자.
    - 두 반사실적 리스크가 같았기 때문이다.
    - <img src="https://render.githubusercontent.com/render/math?math=Pr[Y^{a=1}=1] = Pr[Y^{a=0}=1] = 0.5">
    - 인과 무효(Causal null)를 표현하는 방법 : 두 값의 차이가 0, 또는 두 값의 비가 1
    <img src="https://github.com/DoyoungKim12/causal-inference/blob/master/image_CH1/capture_1.PNG?raw=true">

        - 각각의 명칭은 causal risk difference, risk ratio, odds ratio이다.
        - 위 3개의 지표들은 같은 인과효과의 크기를 다른 스케일로 보여준다.
        - 우리는 이제 위와 같은 지표들을 효과 측정치(effect measures)라 칭한다.
            - 추론의 목적에 맞는 효과 측정치를 사용해야 한다. (multiplicative scale vs. additive scale)

<br>

### 1.4 Random variability

### 무작위 변동성

- 보통의 현실에서 우리가 얻을 수 있는 것은 모집단의 샘플 뿐이다.
    - 따라서, 우리는 처리에 따른 정확한 결과를 알 수 없다. (추정할 뿐이다.)

- 위의 제우스 family 예제로 다시 돌아가보자.
    - 저 20명이 전체 모집단이 아닌, 훨씬 더 큰 집단의 샘플이라고 가정하자.
    - 이제 각 반사실적 결과에 대한 확률은 추정치로 기능하게 된다.
        - <img src="https://render.githubusercontent.com/render/math?math=\widehat{Pr}[Y^{a=0}=1] = 0.5"> 는 <img src="https://render.githubusercontent.com/render/math?math=Pr[Y^{a=0}=1]">의 일치 추정량(consistent estimator)이다.
        - 왜냐하면, 샘플의 수가 커질수록 실제 값과 추정량의 차이는 작아지기 때문이다.
        - 샘플링 변동성(sampling variability) 때문에 발생하는 에러는 무작위로 발생하고, 이는 큰 수의 법칙을 따르기 때문에 위처럼 말할 수 있다.
        - 모집단의 확률 <img src="https://render.githubusercontent.com/render/math?math=Pr[Y^{a=0}=1]">은 계산할 수 없기 때문에 샘플의 확률로 추정하되, 이를 평가하기 위해 통계적 절차가 사용된다.

- 랜덤 에러로 인한 샘플링의 변동성만을 살펴봤지만, 변동성에는 다른 원인도 존재할 수 있다.
    - 개인의 처리에 대한 결과가 확정적이지 않고, 확률적인 경우(non-deterministic counterfactual)를 고려해야 한다.
    - 관찰한 샘플들의 결과가 "동전던지기의 결과"일수도 있는 것이다.
    - 게다가, 개인마다 처리에 대한 결과가 발생할 확률이 다 다를 수 있다. (양자역학은 확정적이지 않은 결과에 대한 좋은 예이다.)

- 따라서, 인과추론에서 랜덤 에러는 두 개의 이유로 발생한다.
    - sampling variability, non-deterministic counterfactual, or both.
    - 하지만, (일단은 직관적 이해를 위해) **이러한 랜덤 에러는 10장까지는 무시**할 것이다.

<br>

### 1.5 Causation versus association

### 원인 VS. 연관

- 우리가 실제로 관찰하게 되는 데이터는 1.2의 예제와는 다르다.
    - treatment가 0일 때와 1일 때를 모두 관찰하는 것은 불가능하다.
    - 우리가 아는 것은 treatment level과 outcome뿐이다.
    - 실제 우리가 볼 수 있는 데이터의 예시
    <img src="https://github.com/DoyoungKim12/causal-inference/blob/master/image_CH1/capture_2.PNG?raw=true">

- 독립(independence)
    - 조건부 확률로 보았을 때, 조건의 변화에도 확률의 변화가 없으면 조건과 결과는 독립이다.
    - <img src="https://render.githubusercontent.com/render/math?math=Y \coprod A">로 표기한다.
        - <img src="https://render.githubusercontent.com/render/math?math=Pr[Y=1|A=1] - Pr[Y=1|A=0] = 0">를 뜻한다.
        - 위처럼 risk difference가 0이다, 또는 risk ratio와 odds ratio가 1이다 로 표현할 수도 있다.

- 연관(association)
    - 독립이 아닌 경우
    - <img src="https://render.githubusercontent.com/render/math?math=Pr[Y=1|A=1] \neq Pr[Y=1|A=0]">
    - treatment와 outcome이 **연관**되어있다면 risk difference, risk ratio, odds ratio는 연관도로 기능한다(association measures).
    - 이 연관도 역시 무작위 변동성의 영향을 받지만, 역시 10장 전까지는 무시한다.
    
- Causation과 association의 차이?
    - Causation : 전체 20명에 대하여, 20명 모두가 이식을 받았을 때와 안받았을 때의 사망 위험도를 계산하여 비교
        - “what would be the risk if everybody had been treated?”
        - 실제로는 관찰할 수 없는 것...
        - 가상의 데이터에서는 둘 다 0.5로 같아 인과관계가 없다고 보았다.
    - association : 전체 20명에 대하여, 각각 이식을 받은 사람과 그렇지 않은 사람의 사망 위험도를 계산했다.
        - “what is the risk in the treated?”
        - 실제로 관찰할 수 있는 것!
        - 7/13 과 3/7 은 그 값이 달라, 연관성이 존재한다고 말할 수 있다.

- 이제 'effect'는 단순 연관을, 'causal effect'는 인과관계를 뜻하는 것으로 한다.
    - well-known adage : “association is not causation.”
    - 그렇다면, 위의 예시처럼 연관성은 있되, 인과관계는 존재하지 않는 상황을 어떻게 해석해야 할까?
        - 심장이식을 받은 사람들의 사망률이 높은 것은, 심장이식 대상자의 건강상태가 더 좋지 않아서 일수도 있다!
        - 우리는 7장에서 **이러한 연관과 인과의 차이(discrepancy)를 교란(confounding)이라 칭할 것이다.**

- 그렇다면 어떤 조건에서 실제 데이터를 인과추론에 사용할 수 있을까?
    - 2장에서의 무작위 시험(randomized experiment)을 통해 가능하다!
        
<br><br><br><br>

## Chapter 2. Randomized experiments

## 무작위 시험
 - 이 장에서는 확률화(Randomization)가 어떻게 인과추론을 설득력있게 만드는지 설명할 것이다.
    - 랜덤 시행 vs. 결정론적(deterministic) 시행
        - **무언가에 의해** 시행이 결정된다면, 그것이 결과 해석에 영향을 끼치게 된다.
        - 따라서, 랜덤하게 시행한 결과를 얻는 것이 중요하다.

<br>

### 2.1. Randomization

### 확률화

- 이전 장에서 논의했듯, 두 잠재적 결과를 모두 관찰할 수 없다.
    - 그렇기 때문에, 연관성만을 측정할 수 있다.
    - 아래 그림의 물음표가 다 채워져야 인과효과를 측정할 수 있다.
    <img src="https://github.com/DoyoungKim12/causal-inference/blob/master/image_CH1/capture_3.PNG?raw=true">
    
    - 무작위 시험 역시 위와 같이 다른 잠재적 결과들을 결측치(missing value)로 남긴 채로 데이터를 생성하게 된다.
        - 하지만, 확률화는 이러한 결측치가 **우연히** 발생했다는 점을 보증한다.
        - 따라서, 인과효과 측정이 가능하다. (정확히는, 일치추정량을 계산하는 것이 가능하다.)

    - 교체성(Exchangeability)
        - 예를 들어, 특정 집단을 동전 던지기로 2개 그룹으로 나누었다고 가정해보자. (흰색 그룹 & 회색 그룹)
        - 이 때의 교체성이란, 흰색 그룹과 회색 그룹 중 어떤 그룹에 처리(treatment)가 가해지더라도 결과가 동일한 것을 뜻한다.
            - 즉, 흰색 그룹에 처리를 가하고 회색 그룹을 대조군으로 두든, 그 반대로 하든 조건부 확률(Risk)이 같다는 말이다.
            - <img src="https://render.githubusercontent.com/render/math?math=Pr[Y^{a}=1|A=1]">은 a에 관계없이 같다.
    
    - 확률화(Randomization)는 바로 이 교체성을 만족시켜 주기에 중요하다.
        - '동전 던지기'라는 무작위 샘플링을 통해, 교체성을 만족하는 두 서브그룹을 만들었다. 
        - 처리가 외생성(exogeneity)을 가진다는 말은 교체성을 갖는다와 동일한 뜻이다.
        
    - 교체성을 통해, 무엇을 얻을 수 있는가?
        - 교체성이 만족된다면, 특정 그룹의 결과는 전체의 결과와 같다.
            - 어느 그룹에 처리가 이루어지든 조건부 확률이 같다면, 전체 그룹의 확률 또한 같기 때문이다.
            - 따라서, 흰색 그룹에/처리를 가했을 때/Y가 1일 확률은 (전체 그룹에)/처리를 가했을 때/Y가 1일 확률 과 같은 의미가 된다.
            - 수식으로 표현하자면, <img src="https://render.githubusercontent.com/render/math?math=Pr[Y=1|A=1] = Pr[Y^{a=1}=1]"> 
            - 마찬가지로, (전체 그룹에)/처리를 가하지 않았을 때/Y가 1일 확률도 위처럼 구할 수 있다. (회색 그룹에서의 risk)
            - **In ideal randomized experiments, association is causation.**
            
    - 교체성과 독립성의 차이를 이해해야 한다.
        - 교체성이 성립한다고 독립성이 성립하는 것이 아니다.
        - 교체성이 성립하는 상황에서 독립성이 성립하지 않는다. = 처리와 결과 사이에 인과관계가 존재한다.






