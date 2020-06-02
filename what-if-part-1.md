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

<br>

### 2.2. Conditional randomization

### 조건부 무작위

<img src="https://github.com/DoyoungKim12/causal-inference/blob/master/image_CH1/capture_4.PNG?raw=true">

- 위와 같이, 기존의 테이블에 L이라는 변수가 하나 추가되었다.
    - L은 예후를 뜻한다. 즉, L이 1이면 컨디션이 나쁜 것으로 본다.
    - 이제 위의 테이블로 상호배타적인 두 실험 디자인에 대해 살펴볼 것이다.
    
- 2개의 랜덤 디자인
    - 디자인 1 : L값에 관계없이 65%의 확률로 처리를 가한다.
    - 디자인 2 : L값에 따라 그룹을 나누어, L이 1인 경우 75%의 확률로 처리 시행 + L이 0인 경우 50%의 확률로 처리 시행
    - 위 테이블은 어떤 실험 디자인을 따르든 가능한 결과이다.
    
- 둘 다 무작위 실험이 맞지만, 조건에 따라 다른 확률을 부여하는지의 여부가 다르다.
    - 디자인 1은 주변부 무작위 실험(marginally randomized experiments), marginal = unconditional이다.
    - 디자인 2는 **조건부 무작위 실험(conditionally randomized experiments)**
    - 디자인 상의 문제로, 디자인 2에서는 교체성이 성립하지 않는다(상대적으로 처리를 시행한 쪽의 예후가 더 안좋기 때문)
    
- 그럼 조건부 무작위로는 교체성을 전혀 확보할 수 없는가?
    - 그건 아니다. 조건 하에서는 무작위로 집단을 구분했기 때문에, 같은 L값을 가지는 서브그룹끼리는 교체성이 성립한다.
    - 이것을 **조건부 교체성**이라 칭한다.
    - 따라서, **무작위 선택(randomization)은 교체성 혹은 조건부 교체성을 항상 보장**한다. 
    
- 그럼 인과위험률(causal risk ratio)는 조건부 무작위 하에서 어떻게 구하는가?
    - 두 가지 방법이 있다.
    - 첫번째 : 계층별 인과위험률(stratum-specific causal risk ratio)을 구한다.
        - 수식 : <img src="https://render.githubusercontent.com/render/math?math=Pr[Y=1|L=1, A=1]/Pr[Y=1|L=1, A=0]"> 
        - 같은 계층이라면(L=1), 계층 내 서브그룹은 교체성이 성립하여 각 그룹내 risk가 전체 risk의 추정치로 기능할 수 있다.
        - 만약, L값에 따라 위험률이 다르게 측정된다면 **L에 의한 효과 변경(effect modification)이 발생**했다고 말한다.
    - 두번째 : 지금까지 해왔던 것처럼 평균인과효과를 계산한다.
        - 어떻게 조건부 무작위 실험에서 평균인과효과를 계산할까?
        - 다음 챕터에서 살펴보자!

<br>

### 2.3. Standardization

### 표준화

- 계층 내 서브그룹에서 교체성(conditional exchangeability)이 성립한다면, 계층별 인과위험률의 **가중평균**으로 인과효과(risk)를 계산할 수 있다.
    - 가중평균으로 전체 효과를 계산하는 방식을 보통 **표준화**라고 한다.
    - <img src="https://render.githubusercontent.com/render/math?math=Pr[Y^{a=1}=1] = \sum Pr[Y=1|L=l, A=1]Pr[L=l]"> 
    - 이처럼, (관찰불가능한) 반사실적 효과의 크기를 **관찰가능한** 데이터의 분포(확률) 함수로 표현할 수 있다면
        - 그 반사실적 효과가 **식별되었다(identified)** 또는 식별 가능하다 라고 표현한다.

<br>

### 2.4. Inverse probability weighting (IP weighting)

### 역확률 가중치

- 계층 내 서브그룹에서 교체성(conditional exchangeability)이 성립한다면, 각 결과(Y)에 **역확률 가중치를 곱한 값**으로 인과효과(risk)를 계산할 수 있다.
    - **결과적으로는 표준화(가중평균)과 같은 방법**이지만, 간단히 개념만 이해하고 넘어가자.
    - 조건부 교체성이 성립한다면, 특정 서브그룹은 해당 그룹의 결과를 대변할 수 있게 된다.
    - 따라서, 각 결과값의 count는 각 그룹의 전체크기에 비례하게 불려서 이해해도 무관하다.
        - 예를 들어, L=1인, 즉 예후가 좋지 않은 12명 그룹에서 전체의 25%인 3명만 떼어내어 처리를 하지 않았다고 하자
        - Y=0, 즉 산 사람이 1명, 반대로 죽은 사람이 2명이다 & 조건부 교체성이 성립하여 나머지 75%를 포함한 결과 또한 대변할 수 있다
        - 위와 같은 경우라면 3명의 결과를 12명의 결과로 불려서 말할 수 있다. (1/0.25 = 4배의 가중치를 각 count에 부여하여 계산할 수 있다)
        - 아래의 Tree diagram으로 이해할 수 있다.
        <img src="https://github.com/DoyoungKim12/causal-inference/blob/master/image_CH1/capture_5.PNG?raw=true">
        
- 이제 우리는 조건부 교체성이 성립할 때의 평균인과효과를 구할 수 있다.
    - 그런데 이 책이 아직 끝나지 않은 이유는?
    - 완벽한 무작위 실험이란 세상에 존재할 수 없기 때문이다. (비윤리적이며 실현불가능한 경우가 많다.)
        - 예를 들어, 심장이식의 경우 더 필요한 사람에게 장기가 이식되기 마련이다. (랜덤으로 배정되진 않는다.)
        - 이러한 상황이라면, 보통 관찰연구(observational study)를 수행하는 것이 그나마 낫다.

<br><br><br><br>

## Chapter 3. OBSERVATIONAL STUDIES

## 관찰연구
- “does one’s looking up at the sky make other pedestrians look up too?” 실험에 대해 다시 생각해보자.
    - 이 가설을 검증하기 위해 내가 매번 고개를 들었다 내렸다 하는 일은 힘이 든다.
    - 따라서, 2명의 행인 그룹을 관찰하여 결과를 계산하려고 한다. (첫번째 행인의 고개들기 여부, 두번째 행인의 행동)
    - 이처럼 실험자가 관련 데이터를 관찰하고 기록하는 형태의 과학실험을 **관찰연구**라고 한다.
- 이 장에서는 관찰연구를 인과추론으로 이끄는 몇 가지 조건에 대해 소개하겠다.

<br>

### 3.1. Identifiability conditions

### 식별가능성 조건

- 관찰연구로부터 인과추론을 하기 위해서는 2가지 재료가 필요하다 : 데이터와 식별가능성 조건들
    - 식별가능성 조건은 총 3가지이다.
        - 각각 일치성, 교체성, 양의 가능성과 대응되며, 이에 대해 뒤에서 다룰 것이다.
    - 식별가능성 조건을 만족하지 않는 경우, 도구변수를 사용하는 방법이 있다. (CH 16에서 다룰 것)

<br>

### 3.2. Exchangeability 

### 교체성

- 관찰연구에서 조건부 교체성이 성립한다는 것을 증명할 방법은 없다.
    - 아무리 많은 변수(outcome predictor)를 고려한다고 해도, 고려하지 못하는 변수는 반드시 존재한다고 생각해야 한다.
    - 따라서, 특정 변수의 분포가 그룹마다 다를 수 있고, 그 변수가 결과에 영향을 미친다면 조건부 교체성은 성립하지 않을 수 있는 것이다.
    - 그래서 결국 교체성은 성립하는 것이 아니라 **가정하는 것이다.**
        - 가정이 더 그럴싸하기 위해서는 아무래도 여러 변수를 탐색하고 전문지식을 활용하는 것이 좋겠다.
        
<br>

### 3.3. Positivity 

### 양의 가능성

- 양의 가능성이란, 특정 조건 하에서 특정 처리를 받을 확률이 모두 0보다 크다는 것을 뜻한다.
    - 즉, 모든 가능성이 열려있어야 한다는 조건이다.
    - <img src="https://render.githubusercontent.com/render/math?math=Pr[A=a|L=l] > 0"> for all values l 

- 양의 가능성 또한 관찰연구에서 성립한다고 보장할 수 없다.
    - 예를 들어, 의사가 예후가 좋지 않은 모든 환자에게 무조건 처방할 수도 있는 것이다.
    - 이런 경우, 예후가 좋지 않은 사람이 처방받지 않을 확률은 0이 되어 양의 가능성이 성립하지 않는다.
    - 양의 가능성이 교체성과 다른 점은, 양의 가능성이 경험적으로 검증될 수는 있다는 것이다.

<br>

### 3.4. Consistency: First, define the counterfactual outcome 

### 일치성 : 먼저 반사실적 결과를 정의하라.

- 일치성이란, 처리에 따른 결과가 제대로 나왔다는 것을 말한다.
    - 예를 들어, 어떤 사람이 처리를 받았다면 그 결과는 처리를 행했을 때의 결과이어야만 한다.
    - 누군가를 하루 종일 앉지 못하게 했다면, 다리가 아파야 정상인 것이다.
    
- 너무나 당연한 조건이라, 이 조건이 성립하지 않는 경우가 있긴 한 것인지 의심스러울 것이다.
    - 하지만, 현실에서 '처리'라는 것이 완벽히 동일할 수 있을까?
    - 심장이식을 시행하는 의사가 여러 명이라면, 심장이식이라는 '시행'의 version은 다르다고 볼 수도 있는 것이다.
    - 아스피린을 투약하는 처리와 사망이라는 결과를 관찰했다면, 아스피린을 투약하지 않았을 때의 결과는 확실히 다른가?
 
 - 그래서 일단 반사실적 결과(관찰되지 않는 평행우주의 결과)를 잘 정의할 필요가 있다.
    - 단순히 '비만'을 처리로 두는 것은 잘못된 것이다.
        - '비만'이 되는 원인은 여러가지이기 때문에, 실제로 비만인 사람이 사망했다고 하더라도 비만 자체는 원인이 아닐 수 있다.
        - '비만이 아닌 사람'이 되는 원인 또한 여러가지이기 때문에, 실제로 비만이 아닌 사람이 생존했다고 하더라도 비만이 아닌 것 자체는 원인이 아닐 수 있다.
    - 보다 구체적인 version이 필요한 것이다. (비만의 원인, 강도, 종류 등...)
    - 그렇다고 눈동자의 색깔과 같이 의미없는 것들까지 고려할 필요는 없겠다. (의미있는 모호함만을 제거하면 충분하다)
    - 위와 같은 것들을 고려하여 잘 정의된 개입(처리)를 **sufficiently well-defined interventions**라고 표현한다.
        
-  그렇다면 우리가 정의한 처리가 sufficiently well-defined인지를 어떻게 알 수 있을까?
    - 알 수 없다.
    - 모호함의 영역을 줄여나갈 수야 있겠지만, 완전히 없애는 것은 불가능할 것이다.
    - 따라서, 인과관계에 대한 질문은 결국 도메인 지식과 주관적 판단에 달렸다.
    - 질문을 구체화해나갈수록 우리가 원래 알고자하는 것과는 멀어지겠지만, 어쩔 수 없다.
    
<br>

### 3.5 Consistency: Second, link counterfactuals to the observed data

### 일치성 : 다음으로, 반사실적 결과를 관찰된 데이터와 연결시켜라

- 우리가 정의한 처리가 sufficiently well-defined되었다고 가정하자.
    - 문제는 현실에 비교가능한 데이터가 없을 때가 있다.
    - 처리 자체를 지나치게 구체적으로 설정하는 경우, 이런 문제가 발생한다.

- 해결방법 중 하나는 모든 처리 version이 같은 효과를 가져온다고 가정해버리는 것이다.
    - **treatment-variation irrelevance**라고 표현한다.
    - 실제로 모든 version을 구분하는 것이 불가능하기 때문에 암시적으로 이처럼 가정할 수밖에 없다.
    
- 최선의 방법은 treatment-variation irrelevance 가정을 가능한 한 명료하게 하는 것이다.
    - 다음 섹션에서 그 명료함에 도달하는 방법을 탐색할 것이다.
    
<br>

### 3.6 The target trial
    
### 타겟 시험

- 인과효과 측정을 위한 무작위 시험 = 가상 실험(hypothetical experiment) = 타겟 시험(target experiment or the target trial)
    - 타겟 시험이 윤리, 시간 등의 이유로 불가능할 경우, 우리는 관찰된 데이터를 통해 이를 **모방(emulation)**하는 것이다.

- “what randomized experiment are you trying to emulate?”
    - 이것이 관찰된 데이터로 수행하는 인과추론의 핵심 질문이 된다.
        - 첫째로, 우리가 수행하고 싶지만 불가능한 타겟시험이 무엇인지?
        - 둘째로, 관찰된 데이터만으로 어떻게 타겟시험을 모방할 것인지?
        
- 타겟 시험이 무엇인지를 정의하기 위한 핵심 구성요소(key components)들이 있다.
    - eligibility criteria(자격 기준), interventions(개입,**처리**), outcome, follow-up, causal contrast, and statistical analysis.
    - 이 챕터에서는 그룹 간 비교할 **처리**에 집중할 것이다.
    
- 잘 정의된 개입(**sufficiently well-defined interventions**) = 타겟 시험의 명료한 모방(explicit emulation)
    - 그러나 모든 사람들이 이에 동의하진 않음.
        - 예를 들어, 비만과 사망의 인과관계를 파악하는 문제에서 비만이라는 개입이 구체적으로 정의되지 않았다고 하자.
        - 그렇다고 비만이 사망에 영향을 미치지 않았다고 말할 수 있는가?
        - 비만이 사망에 영향을 준다는 '가능성'을 제시하는 것만으로도 충분한 의미가 있을수 있다.
    - 이러한 반대 주장은 나름 일리가 있으나, 위험한 생각이다.
        - 교체성, 양의 가능성에 대한 올바른 고려가 없는 추론은 위험하다.
        - 개입이 구체적으로 잘 정의되지 않으면 조건부 교체성에 대한 불확실성이 지나치게 커진다.
        - 마찬가지의 이유로 양의 가능성이 성립하지 않은 가능성도 지나치게 커진다. (특정 특성을 가진 집단이 모두 개입=1 일 수 있는 것이다.)
        - 이러한 불명확한 개입(처리)로 인해 발생하는 문제는 통계적 기법으로 해결 가능한 문제가 아니라서 더 위험하다.
        
- 물론, 연관성을 관찰하는 것도 의미가 있다.
    - 하지만, 어디까지나 가설 생성을 위한 과정일 뿐이다.
    - 연관성만을 가지고 의사결정을 하긴 어렵지 않은가.

<br><br><br><br>

## Chapter 4. EFFECT MODIFICATION

## 효과 수정

- 인과효과는 특정 집단의 특성에 따라 달라질 수 있다.
    - 집단 전체에 적용되는 처리에 대한 연구라면 전체 집단에 대한 ATE가 필요하겠지만,
    - 서브그룹 간 차이를 관찰한다면 서브그룹별 ATE를 산출해서 비교할 수도 있는 것이다.
    
<br>

### 4.1 Definition of effect modification
    
### 효과 수정의 정의

- 성별에 따른 서브그룹별 인과효과를 살펴보자
    - <img src="https://github.com/DoyoungKim12/causal-inference/blob/master/image_CH1/capture_6.PNG?raw=true">
    - V는 성별을 나타내는 것으로, V=1이면 여자 / V=0이면 남자이다.
    - 전체로 보면 리스크 차이가 없어 인과효과가 없지만, V로 구분한 서브그룹별로는 인과효과가 존재한다.
    
- 이제부터 V를 수정자(modifier)라고 하겠다.
    - V의 level에 따라 A의 Y에 대한 인과효과가 다르게 측정될 때, V를 A의 Y에 대한 효과의 수정자라고 한다.
    - 효과 수정은 어떤 기준으로 효과를 측정했냐에 따라 다르게 나타날 수 있다.
        - 위험차(risk difference) 기준 효과조정 측정 : Additive effect modification
        - <img src="https://render.githubusercontent.com/render/math?math=E[Y^{a=1} - Y^{a=0}|v=1] \neq E[Y^{a=1} - Y^{a=0}|v=0]">
        - 위험비(risk ratio) 기준 효과조정 측정 : Multiplicative effect modification
        - <img src="https://render.githubusercontent.com/render/math?math=\frac{E[Y^{a=1}|v=1]}{E[Y^{a=0}|v=1]} \neq \frac{E[Y^{a=1}|v=0]}{E[Y^{a=0}|v=0]}">
        
- 어떤 measure로 측정했느냐에 따라 효과 수정이 포착될 수도 있고, 그렇지 않을 수도 있다.
    - 이러한 문제점 때문에, 질적 효과 수정(qualitative effect modification)을 효과-측정 수정(effect-measure modification)으로 표기하기도 한다.
    - 질적 효과 수정(qualitative effect modification) : 집단 간 인과효과의 방향(음 또는 양)이 다를 때를 말한다.
        - 어떤 집단에서는 처리와 결과가 양의 관계일 수 있고, 어떤 집단에서는 반대라면 인과효과가 '질적으로' 다른 것이다.
    - Multiplicative이지만 not Additive인 경우도 발생할 수 있다.
    
<br>

### 4.2 Stratification to identify effect modification
    
### 효과 수정을 식별하기 위한 계층화

- 효과 수정을 식별하기 위한 자연스러운 방법 : 계층화 분석(stratified analysis)
    - 4.1의 table은, 당연하지만 실제로 관찰가능한 데이터가 아니다.
    - 실제 관찰가능한 데이터를 보자
      <img src="https://github.com/DoyoungKim12/causal-inference/blob/master/image_CH1/capture_7.PNG?raw=true">
    - 반사실적 결과의 관찰불가능성이 효과수정을 판별하기 위한 계층화에 어떻게 영향을 미치는 것일까?
        - 그 답은 연구 설계에 달렸다.
        
- 계층화 분석은 간단하다. 
    - 지난 2장에서 예후(L)에 따라 표준화된 효과측정치를 구했던 것을 기억할 것이다.
    - 이처럼 표준화된 측정치를 각 계층(V)별로 구하여 비교하면 된다.
    - 위험차 또는 위험비에 차이가 있다면, 효과 수정이 있다고 말할 수 있다.
        - 그러나, 각 측정치에 차이가 있더라도 어쨌든 효과의 방향이 같다면 질적 효과수정은 없다고 말할 수 있다.
        
- 그런데, V가 사실 S라는 실제 수정자의 표상(대리)라면 어떨까?
    - 예를 들어, 국적(V)에 따른 심장이식 사망률에 유의미한 차이가 있었다고 하자.
    - 그런데, 이는 사실 국가별 심장이식 의료수준 차이(S)때문에 발생한 것이다.
    - 그렇다면, 국적(V)는 대리효과수정자(surrogate effect modifier) 이고, 의료수준차(S)는 인과효과수정자(causal effect modifier) 라고 말한다.
    - 어쨌든, V가 무조건 효과수정을 가져온다고 말할 수는 없는 것이다.

<br>

### 4.3 Why care about effect modification
    
### 효과 수정에 신경을 쓰는 이유?

- 첫째로, 일반적으로는 세상에 '결과 Y에 대한 처리 A의 평균 인과효과'같은 것은 없다.
    - '**인과효과 수정자들의 특정 조합으로 이루어진 어떤 집단내에서의** 결과 Y에 대한 처리 A의 평균 인과효과'를 관찰하는 것이다.
    - 따라서, 다른 인과효과 수정자 조합으로 이루어진 집단에 실험결과를 그대로 적용할 수는 없는 것이다.
    - 만약 그런 일이 가능하다면, 이동성(transportability)이 존재한다고 말한다.
    - 이는 한 집단 내 서브그룹의 교체성 확인보다 더 어려운 문제이다.
        - 도메인 전문 지식을 활용하여 이동성을 정당화하는 방법밖에 없다.
        
- 둘째로, 효과 수정자를 찾는 것은 개입으로 인해 가장 큰 효과를 받을 서브그룹을 특정하는 데에 도움을 준다.
- 마지막으로, 효과 수정자를 특정하는 것은 결과를 이끌어내는 메커니즘을 이해하는 데에 도움을 준다. 
    - 효과 수정자 특정은 두 처리 간의 상호작용(interaction)을 특정짓는 첫걸음이 될 수 있다.
    - 효과 수정과 상호작용은 분명히 다르며, 이는 5장에서 설명할 것이다. 
    
<br>

### 4.4 Stratification as a form of adjustment
    
### 계층화 (조정 관점)

- 국적, 예후, 처리, 결과 의 4개 컬럼이 있는 테이블을 떠올려보자.
- 이 챕터의 목적은 변수 V(국적)에 의한 효과 수정을 식별하는 것이다. (stratification)
    - **standardization** : 변수 L(예후)에 따라 인과효과의 강도를 표준화
    - **stratification** : 변수 V(국적)에 의한 효과 수정을 식별
    - 하지만, 현실에서는 stratification이 standardization의 대안으로 쓰인다. (많은 연구자들이 stratification과 adjustment를 동일어로 생각)
    - 사실 변수 L로 서브그룹을 만들어서 평균인과효과를 측정하면, 그것이 조건부 효과 측정치가 된다.
    
- 계층화를 조정의 관점에서 사용하게 되면, 
    - 내가 효과 수정에 관심있는지의 여부와 상관없이
    - **조건부 교체성을 달성하는 데에 필요**한 **모든 변수들에 대한 효과 수정을 관찰**할 수 있게 된다!

- 반대로, 표준화 이후 계층화를 사용하게 되면,
    - 교체성과 효과 수정을 분리해서 생각할 수 있게 된다. (위에서 계속 다뤄왔던 내용)

- 제한(restriction)의 관점에서도 계층화를 사용할 수 있다.
    - 양의 가능성을 만족하지 않는 계층(서브그룹)은 분석에서 제외한다.
    
<br>

### 4.5 Matching as another form of adjustment
    
### 매칭 (조정의 또다른 형태)

- 매칭(Matching) : 모든 서브그룹에서 변수 L의 분포가 같도록 구성하는 것
    - 먼저, L이 같은 서브그룹에서 처리 유무에 따라 Pair를 만든다.
        - 예를 들어, L=0 인 그룹에서 처리 A가 1인 사람이 8명, 0인 사람이 4명이라면 4개의 Pair가 만들어진다. (A=1인 나머지 4명의 데이터는 버림)
    - 이렇게 하면 A가 1이든 0이든 L의 분포는 같아질 수밖에 없다.
    - 조건부 교체성 하에서 매칭을 적용하면, 무조건적인 교체성이 성립하여 효과측정을 바로 시도할 수 있다.
    - 매칭이 반드시 1대1(matching pair)일 필요는 없다. (1대다(matching set) 또한 가능)
        - 상대적으로 수가 더 적은 서브그룹이 다른 서브그룹의 수를 규정

<br>

### 4.6 Effect modification and adjustment methods
    
### 효과 수정과 조정 도구

- 평균인과효과를 측정하는 다양한 방법 : **Standardization, IP weighting, stratification/restriction, and matching** 
    - 그러나 각각 다른 타입의 인과효과를 측정함.
    - 위의 네가지 방법은 두 그룹으로 나눌 수 있음.
    <br><br>
        - Standardization, IP weighting : 한계효과와 조건부효과 모두 측정가능
            - 특성별로 그룹을 나눔 (각각 L이 0,1인 그룹)
            - 각 서브그룹 내 outcome의 비율(위험률)을 계산 (예후 L과 처리 A의 조합별 위험율 4개)
            - L에 관계없이, 처리 A가 동일한 그룹의 위험률 값들의 가중평균으로 **처리별 위험률**을 구함
            - 처리별 위험률의 비(ratio)로 인과효과를 산출
        <br><br>
        - stratification/restriction, and matching : 특정 서브그룹의 조건부효과만 측정 가능
            - 특성별로 그룹을 나눔 (각각 L이 0,1인 그룹)
            - 특성별 서브그룹 내에서만 처리별 위험률의 비를 계산 
            - 각 처리별 위험률의 비를 서브그룹끼리 비교하여 효과수정을 관찰하는 방법
   <br><br>         
    - 네가지 방법 모두 교체성과 양의 가능성이 요구된다.
    
- 효과 수정이 없다면, 이 네가지 접근방법으로 계산된 효과 측정치는 모두 동일하다.

<img src="https://github.com/DoyoungKim12/causal-inference/blob/master/image_CH1/capture_8.PNG?raw=true">

- 위 표에서 네가지 측정 도구로 인과효과를 계산해보자. (독자의 몫 = 나의 몫)
    - Standardization, IP weighting : 0.8
    - stratification(L에 따른 서브그룹별 인과효과) : L=1일때 0.5, L=0일때 2.0
    - matching(임의의 pair를 사용) : 1.0 
        - Rheia-Hestia, Kronos-Poseidon, Demeter-Hera, Hades-Zeus for L = 0
        - Artemis-Ares, Apollo-Aphrodite, Leto-Hermes for L = 1
    - 위의 사례는 전체 그룹 또는 서브그룹을 특정하는 일이 얼마나 중요한지 잘 보여준다.
    - 또한, 측정 도구별로 추정치가 다르다고 하여 특정 추정치가 다른 것에 비해 더 잘 맞는다고 말할 수 있는 것은 아니다.
    
- 지난 챕터에서는 의미있는 인과추론의 첫번째 선제조건으로 잘 정의된 인과효과가 필요하다고 말했다.
    - 그리고 이번 챕터에서는 분명히 구분된 타겟 그룹이 필요하다고 주장하였다.
    - 이 두 가지 선제 조건은 정말 중요하며, 실험자가 철저히 증명해내야 할 부분이다. 

<br><br><br><br>

## Chapter 5. INTERACTION

## 교호 작용

- 지금까지는 하나의 처리에 대한 인과효과만을 고려해왔다.
    - 예를 들어, '누군가 하늘을 올려다본다'라는 행위가 다른 사람의 '하늘을 올려다본다'라는 행위에 영향을 주는가에 대한 것이었다.
    - 그러나, 많은 인과적 질문들은 2개 또는 그 이상의 동시다발적 처리에 대한 것이다.
    - 예를 들어, '누군가 하늘을 올려다 볼 때', 그 누군가가 '옷을 입고 있는지 또는 벗고 있는지'도 고려할 수 있다.
    
- 만약 두 경우의 인과효과가 다르다면, 우리는 두 처리(올려다보기 여부, 옷입기 여부)가 결과를 도출하는 과정에서 **상호(교호) 작용하였다**고 말할 수 있다.

- 여러 개의 처리를 조합한 joint intervention(결합 개입)이 가능하다면, 교호작용의 식별은 가장 효과적인 개입이 무엇인지 밝혀줄 것이다.
    - 이 챕터에서는 두 처리 간 교호작용의 일반적인 정의를 두 가지 틀 안에서 살펴볼 것이다.
        - 하나는 우리가 이미 알고 있는 반사실적 프레임워크, 다른 하나는 sufficient-component-cause framework 이다.
        
<br>

### 5.1 Interaction requires a joint intervention
    
### 교호작용에는 결합 개입이 필요하다

- 우리가 지금껏 다뤄온 심장이식 예제에서, 개별 피실험자들의 심장이식 여부 이전에 비타민 투약 여부의 처리가 있었다고 가정하자.
    - 비타민 투약 여부는 E로 표기 한다 (E=0, E=1)
    - 2가지 경우의 수를 가지는 2개 처리가 있으므로, 총 4개의 잠재적 조합이 발생한다. 
        - (E=0, A=0), (E=0, A=1), (E=1, A=0), (E=1, A=1)
    - 이처럼 2개 이상 처리의 조합을 **결합 개입** 이라고 칭한다.
    
- 이제 반사실적 프레임워크 안에서 교호작용을 정의할 수 있다.
    - 다른 처리들이 같을 때, 비타민을 투약했을 때와 그렇지 않았을 때의 인과효과가 다르다면 교호작용이 있다고 말할 수 있다.
    - 인과효과가 위험도 차이(risk difference)로 측정될 때, 아래의 조건을 만족하면 A와 E 사이에 교호작용이 존재한다. (E가 고정된 상태에서의 차이) 
        - <img src="https://render.githubusercontent.com/render/math?math=Pr[Y^{a=1,e=1}=1] - Pr[Y^{a=0,e=1}=1] \neq Pr[Y^{a=1,e=0}=1] - Pr[Y^{a=0,e=0}=1">
        - 간단한 식 변환을 통해, E가 아닌 A가 고정된 상태에서의 차이로도 교호작용 식별이 가능함을 알 수 있다.
        - <img src="https://render.githubusercontent.com/render/math?math=Pr[Y^{a=1,e=1}=1] - Pr[Y^{a=1,e=0}=1] \neq Pr[Y^{a=0,e=1}=1] - Pr[Y^{a=0,e=0}=1">
        
- 그렇다면 효과수정(effect modification)과 교호작용(interaction)의 차이는 무엇일까? 
    - 효과수정자 V는 결과 Y에 직접적으로 영향을 미치지 않는다고 가정한 것이다.
        - V의 수준에 따른 A의 인과효과 변화를 보는 것이지(effect modification), V의 수준에 따른 Y의 차이를 보는 것이 아니다.
        - V는 A와 동등한 지위를 갖지 않는 것이다. 
    - 반면, E는 A와 동등한 지위(=처리)를 가지기 때문에 E와 A의 인과효과 변화를 관찰하게 된다(interaction).
    
<br>

### 5.2 Identifying interaction
    
### 교호작용 식별하기

- 이전 챕터에서는 '결과 Y에 대한 처리 A의 평균인과효과'를 식별하기 위한 조건들을 제시했다.
    - 3가지 핵심 조건 : exchangeability, positivity, consistency
    - 교호작용은 2개 이상의 처리를 전제로 하기 때문에, 모든 처리에 대해 위 3개 조건이 만족함을 보여야한다.
    
- 비타민(E)이 피실험자들에게 무작위로, 다른 조건과 관계없이 투약되었다고 가정하자.
    - 위처럼 가정한다면, 3개 조건이 모두 성립한다.
    - 일반적으로 결합확률은 조건부확률과 같으므로 아래와 같이 교호작용을 재정의할 수 있다.
    - <img src="https://render.githubusercontent.com/render/math?math=Pr[Y^{a=1}=1 | E=1] - Pr[Y^{a=0}=1 | E=1] \neq Pr[Y^{a=1}=1 | E=0] - Pr[Y^{a=0}=1 | E=0]">
    - 이처럼 조건부확률로 정의할 경우, 교호작용의 개념은 효과수정의 개념과 정확히 같아진다.
        - 따라서, 위에서 사용한 효과수정 식별 방법을 그대로 교호작용 식별에 적용하면 된다.
        
- 이제 비타민이 통제되지 않은, 즉 무작위성을 담보할 수 없는 처리가 되었다고 가정하자.
    - A와 E 사이의 교호작용 여부를 살펴보기 위해서는 4개의 한계 위험도(marginal risks)을 계산해야 한다.
    - marginal randomization이 전제되지 않았다면, 해당 위험도는 Standardization이나 IP weighting과 같은 일반적인 식별방법을 사용할 수 있다.
        - A와 E의 조합인 4개의 처리가 있다고 가정하고, 나머지는 동일한 과정을 사용하여 풀어낼 수 있다.
        
- A는 식별조건이 성립하고 E는 그렇지 않을 때, E에 따른 서브그룹별로 A의 인과효과를 측정한다면?
    - 효과수정을 식별할 수는 있겠으나, 교호작용은 식별할 수 없다.
        - E에 대해 인과효과 식별조건이 성립하지 않기 때문이다.
        - 효과 수정자는 애초에 3개 조건을 필요로 하지 않는다는 것을 4장에서 보았다.
    - 효과수정자 : 실제로 식별되지는 않았지만, 결과 Y에 영향을 주고 처리 A와 교호작용을 보이는 변수(처리)와 연관된 지표
        - V가 Y에 직접 영향을 끼친다고 가정하지 않는다.
        - 효과수정은 교호작용이 발견되지 않아도 발생할 수 있다. (식별되지 않은 다른 변수(처리)의 영향)

<br>

### 5.3 Counterfactual response types and interaction
    
### 반사실적 반응의 타입과 교호작용

- 하나의 처리가 있다면, 그에 대해 가능한 반응 타입은 4개이다.
    - Doomed : 처리에 관계없이 Y = 1
    - Helped : 처리를 받으면 Y = 0, 그렇지 않을 경우 Y = 1
    - Hurt : 처리를 받으면 오히려 Y = 1, 그렇지 않을 경우 Y = 0
    - Immune : 처리에 관계없이 Y = 0
    
- 그렇다면, 2개의 binary 처리에 대해서는 16개의 반응 타입이 가능하다.
    - 처리의 경우가 4가지, 각 처리에 대한 결과는 0 또는 1로 2가지.
    - 2가 4번 곱해지는 꼴으로 총 16개이다.

<img src="https://github.com/DoyoungKim12/causal-inference/blob/master/image_CH1/capture_9.PNG?raw=true">

- 1, 4, 6, 11, 13, 16번의 경우에 주목해보자
    - 위 6개 타입은 A 또는 E에 관계없이 일정한 결과를 보이는 경우이다.
    - 따라서, 모든 피실험자가 위 6개 타입에 속한다면, A와 E 사이에는 교호작용이 없다고 말할 수 있다.
    
- A와 E 사이에 교호작용이 존재한다는 것은 곧 특정 피실험자에 대해서는 두 반사실적 결과가 하나의 처리 여부로는 결정될 수 없음을 의미한다.
    - 처리 A(E)만으로는 반사실적 결과를 정의할 수 없어, E(A)를 꼭 참고해야 한다는 뜻이다.
    - 아래 3가지 경우중 하나에 속하는 피실험자가 있다는 것을 의미한다.
        - 1. 4개 경우의 수 중 하나에만 반응하는 경우 (8, 12, 14, 15)
        - 2. 두 처리가 서로 반대로 작용하는 경우에만 반응하는 경우 (7, 10)
        - 3. 4개 경우의 수 중 하나에만 반응하지 않는 경우 (2, 3, 5, 9)
        
- 교호작용의 의미는 이처럼 모든 반사실적 반응의 타입을 분류해보는 과정을 통해 보다 명확해졌다.

<br>

### 5.4 Sufficient causes
    
### (특정 결과가 반드시 발생하기 위해 필요한) 충분한 조건

- 하나의 처리가 결과를 결정짓는다고 보기는 어렵다.
    - 여러 개의 의미있는 조건들(causes)이 충분히(Sufficient) 모였을 때, 반드시 Y는 0 또는 1이 된다.
        - 예를 들어, '마취에 알러지가 있는 사람이 수술을 받으면 반드시 사망한다'와 같은 경우가 있겠다.
        - '마취에 알러지가 있다'는 일종의 background factor(배경 요인)이다.
    - 이러한 아이디어를 교호작용의 대안적 개념에 적용시켜 시각화해볼 수 있다.
    
- 하나의 처리는 각각 1이 의미있음 / 0이 의미있음 / 상관없음 의 cause로 기능할 수 있다. (3개의 경우의 수)
    - 2개의 처리가 있으니, 3이 2번 곱해져 9개의 경우의 수가 발생한다.
    - A = 1 only, A = 0 only, E = 1 only, E = 0 only, A = 1 and E = 1, A = 1 and E = 0, A = 0 and E = 1, A = 0 and E = 0, and neither A nor E matter. 
    - 이를 시각화하면 아래와 같다. (“the causal pies.”)

<img src="https://github.com/DoyoungKim12/causal-inference/blob/master/image_CH1/capture_10.PNG?raw=true">

<br>

### 5.5 Sufficient cause interaction
    
### 충분한 조건 하의 교호작용

- 반사실적 프레임워크에서의 교호작용은 단순히 반사실적 결과의 대조로 식별되는 것이었다.
    - E=1일 때의 인과효과와 E=0일때의 인과효과가 다른가를 본 것이다.
    - 이 절에서는 여기서 한걸음 더 나아갈 것이다. 
    - 교호작용의 두번째 개념 : sufficient-component-cause framework 
        - 줄여서 Sufficient cause interaction(충분조건 교호작용)로 쓰겠다.
        
- A와 E 사이의 충분조건 교호작용은 충분조건에서 A와 E가 동시에 등장할 때 존재하게 된다.
    - 예를 들어, 특정 배경 요인을 가진 사람에게 E와 A가 동시 적용될 경우 반드시 특정 결과가 나타난다고 하자.
        - 그리고 E 또는 A만 적용될 경우에는 그렇지 않다고 하자.
    - 이런 경우에는 충분조건 교호작용이 해당 특정 배경요인을 지닌 사람에게 존재한다고 말할 수 있다.
    
- 충분조건 교호작용은 서로 시너지로 작용하거나 그 반대일 수 있다.
    - 위의 예시는 시너지이며, 그 반대(E 또는 A만 적용되어야 특정 결과가 나타남)은 길항이다.
    
- 충분조건 교호작용이 (교호작용의) 반사실적 정의와 다른 점은?
    - 두 처리를 포함하는 인과구조에 대해 명시적인 레퍼런스를 제공한다는 점.
    - 도메인 지식이 반드시 필요한 것은 아님.
    
<br>

### 5.6 Counterfactuals or sufficient-component causes?
    
### 반사실적 프레임워크 vs. 충분조건 (어떤 개념을 써야할까?)

- 결론부터 이야기하면 반사실적 프레임워크가 표준이다.
    - 충분조건은 교호작용을 이해하는 과정에서는 좋은 개념이지만, 실제 데이터 분석으로의 적용은 아직이다.

<br><br><br><br>

## Chapter 6. GRAPHICAL REPRESENTATION OF CAUSAL EFFECTS

## 인과효과의 시각적 표현

- 지금까지는 실제로 존재하지 않는 단순한 시나리오를 가정해왔다. 
    - 그러나 현실의 복잡한 문제를 가정하게 된다면, 우리가 변수에 대해 아는 것과 가정하는 것을 명확히 구분할 필요가 있다.
    - 이 챕터에서는 우리가 관심있는 인과 구조의 선결 가정과 정량적 전문 지식을 시각적으로 표현하는 도구에 대해 소개할 것이다.
    - 이러한 시각적 표현은 개념적인 문제를 보다 명료히 하고, 실험자 간의 소통을 원할하게 한다.

<br>

### 6.1 Causal diagrams
    
### 인과 도표

- 이 챕터에서는 인과 도표로 불리게 될 그래프에 대해 설명한다.
    - 여기와 이후의 세 챕터에서는 인과 도표를 통한 문제 개념화에 집중할 것이다.
    
-  Figure 6.1을 보자.
    - <img src="https://github.com/DoyoungKim12/causal-inference/blob/master/image_CH1/capture_11.PNG?raw=true">
    - 3개의 노드(L, A, Y)와 3개의 엣지(화살표)가 있다.
        - 시간은 왼쪽에서 오른쪽으로 흐른다는 관습적 표현이 적용되어, L이 시간 순서상 가장 먼저 발생한 것이다.
        - 화살표는 최소 1명의 개인에 대한 직접적인 인과효과(direct causal effect)를 의미한다.
        - 화살표가 없다는 것은, 단 1명의 개인에게서도 직접적인 인과효과가 없다는 뜻이다.
    - 이런 종류의 인과 도표를 **방향이 있는 비순환 그래프(directed acyclic graphs, DAGs)** 라고 한다.
        - 방향성은 화살표에 방향이 있음을, 비순환은 특정 변수가 스스로의 원인이 될 수 없음을 암시한다.
        - DAGs는 다른 분야에도 적용되므로, 이제부터는 (causal) DAGs로 이해하자.
    - causal DAG를 정의하는 속성
        - causal Markov assumption : 직접적 인과로 이어지지 않은 모든 변수들은 서로에 대해 독립이다.
            - 다시 말해, 관계가 있는 변수들은 모두 그래프에 포함되어야 한다는 것을 암시한다.
        - 예를 들어, L(질병 보유 유무)은 A(수술 여부)의 각 결과로 할당될 확률에 영향을 주기 때문에 그래프에 반드시 포함되어야 한다.
        
- 그래프가 반사실적 접근보다 직관적이라고 생각하지만, 두 접근은 밀접히 연관되어있다.
    - 일반적으로는 그 연관성이 가려져있었으나, 최근의 causal DAG인 SWIG(Single World Intervention Graph)는 두 접근을 매끄럽게 통합하였다.
    - SWIG의 도입은 필수적인 선제조건을 짚고 난 후로 미루겠다.
    
<br>

### 6.2 Causal diagrams and marginal independence
    
### 인과 도표와 한계 독립

- 인과 도표에서 우리는 변수간의 인과관계에 대한 지식만을 사용하여 그렸지만, 흥미롭게도 이는 상관관계(또는 독립)에 대한 정보도 암시한다.
- 아래의 3가지 예시를 통해 두 변수간의 독립과 상관관계에 대해 알아보자.

- 첫번째 예시 : A와 Y 사이에 엣지가 존재하는 경우
    - <img src="https://github.com/DoyoungKim12/causal-inference/blob/master/image_CH1/capture_12.PNG?raw=true">
    - 인과관계는 상관관계를 암시하므로 A와 Y는 독립이 아니다. (상관관계가 있다.)
    - 상관관계는 인과관계와 달리 대칭적 관계이므로, 화살표의 방향과 관계없다.
    
- 두번째 예시 : L이 A와 Y의 공통 원인(common cause)인 경우
    - <img src="https://github.com/DoyoungKim12/causal-inference/blob/master/image_CH1/capture_13.PNG?raw=true">
    - A와 Y 사이에 인과관계가 없다는 것은 알고 있지만, A와 Y 사이에 상관관계가 없다는 것도 참일까?
    - 해당 관계를 모르고 실험을 할 경우를 가정해보자.
        - A=1일 때 L=1일 가능성이 높아, Y도 1일 가능성이 높아졌다고 해보자.
        - 실험자는 A 조건부의 Y=1 확률이 달라, A와 Y 사이에 상관관계가 있다고 할 것이다.
        - 이렇기 때문에 상관관계를 인과관계로 착각해서는 안된다.
        
- 세번째 예시 : A와 Y가 각각 L의 원인이 되는 경우
    - <img src="https://github.com/DoyoungKim12/causal-inference/blob/master/image_CH1/capture_14.PNG?raw=true">
    - L은 공통 효과(common effect)이자 충돌점(collider)이다.
        - 충돌점으로 불리는 이유는 링크인 화살표의 끝이 '충돌'하는 지점이 되기 때문이다.
    - A 조건부의 Y=1일 확률이 같을 것이기에, A와 Y는 독립이다.
    - 이처럼 충돌점은 상관관계의 흐름을 막는 길목이 된다.
    
- 요약 : 두 변수는 하나가 다른 하나의 원인이 되거나, 같은 원인을 공유할 경우 (한계) 연관되어있다. (그렇지 않을 경우, 독립이다.)

<br>

### 6.3 Causal diagrams and conditional independence
    
### 인과 도표와 조건부 독립

- 이제 다시 그림 6.2, 6.3, 6.4로 돌아가보자.
    - 그림 6.2에 따르면, 우리는 A(아스피린)가 Y(심장병)에 대해 인과효과를 가지기 때문에 서로 상관관계를 가진다고 보았다.
    - 이제, 새로운 정보 B(혈소판 집적)를 알게 되었다고 해보자.
        - A는 Y의 위험성에 영향을 주는데, 그 이유는 A가 B를 전반적으로 낮추는 효과가 있기 때문이었다.
        - 이러한 새로운 지식은 아래와 같이 표현될 수 있다. (B가 A와 Y 사이를 중재(mediator)하고 있는 형태)
        - B 주변의 box : B가 특정 level(0 또는 1중 하나)인 집단의 분석을 제한(restriction)하겠다는 의미
        - <img src="https://github.com/DoyoungKim12/causal-inference/blob/master/image_CH1/capture_15.PNG?raw=true">
        
- 이처럼 3번째 변수(B)가 드러났다면, 새로운 질문이 가능하다.
    - B의 level(1 또는 0)별로 관찰했을 때에도 A와 Y 사이의 상관관계가 존재할까?
    - 다시 말하면, 우리가 B에 대한 정보를 이미 알고있음에도 A라는 정보가 Y를 예측하는 데에 도움이 될 것인가?
        - 이 질문을 위해 제한(restriction)이 필요했던 것이다.
        
- B=0, 즉 혈소판 집적 수준이 낮은 사람은 Y, 즉 심장병에 걸릴 위험이 낮다고 한다.
    - 다시 말해, A가 1이든 0이든 우리는 이미 B를 통해 Y를 잘 예측하고 있는 것이다.
    - 사실, 아스피린은 오직 혈소판 집적 수준을 낮출 때에만 Y에 영향을 주는 것이었다.
        - 결국, 개인의 처리 수준(A)에 대한 정보가 Y의 예측에 아무 도움도 주지 못했다.
    - 따라서, B=0인 서브그룹 내에서는 A와 Y가 연관되었다고 말할 수 없다. (B=1인 서브그룹에서도 동일한 결론을 낼 수 있을 것이다.)
    
- A와 Y가 한계적으로는 연관되어 있더라도, B 조건 하에서는 독립이다.
    - 왜냐하면 B의 level이 동일한 그룹 내에서는 A가 1인 그룹과 0인 그룹의 Y 위험성이 동일하기 때문이다.
    - 따라서, B 주변의 box는 상관관계 흐름의 단절을 의미한다. 
    
- 이제 그림 6.3을 다시 보자.
    - 위 상황에서 새롭게 던질 질문 : L의 level별로 보았을 때, A와 Y가 연관되었을까?
    - 이 질문의 표현 또한 위에서와 같이 L 주변의 box로 표현된다.
    - <img src="https://github.com/DoyoungKim12/causal-inference/blob/master/image_CH1/capture_16.PNG?raw=true">
    
- L=1, 즉 비흡연자인 그룹으로만 분석을 한정한다고 가정하자.
    - 이 경우 역시, A(라이터 소지 여부)가 어떤 값이든 Y(폐암)의 위험성을 예측하는 데에 도움을 주지 못할 것이다.
    - 따라서, A와 Y는 L=1인 서브그룹 내에서는 조건부 독립이다.
    - L 주변의 box는 상관관계 흐름의 단절을 의미하게 된다.
    
- 마지막으로, 그림 6.4를 보자
    - A는 유전자 정보, Y는 흡연 여부, L은 심장병 발병여부이다.
    - 지난 시간에 우리는 이미 A와 Y가 독립이라고 결론내렸지만, 두 변수가 L 조건하에서는 조건부 상관관계를 가질 수 있다.
    - 위에서와 마찬가지로, L 주변에 box를 그려 다른 변수들이 L의 조건 하에 있다는 것을 암시한다.
    - <img src="https://github.com/DoyoungKim12/causal-inference/blob/master/image_CH1/capture_17.PNG?raw=true">
    
- L=1, 즉 심장병이 있는 사람들 중 흡연자 Y의 비율은 유전자 정보 A 값이 낮을수록 올라간다고 한다.
    - 따라서, A와 Y 사이에 조건부 상관관계가 존재한다고 말할 수 있다.
    - A와 Y 사이의 상관관계를 인과관계로 착각하는 실수가 발생할 수도 있을 것이다.
    - 아무튼, 충돌점 주변의 box는 상관관계의 흐름을 잇는 역할을 한다.
    
- 직관적으로, 두 변수(원인)가 연관되어있는지의 여부는 미래의 이벤트(결과)에 영향을 주지 못한다.
    - 하지만, 주어진 결과 하의 두 원인은 보통 연관되어있다. (우리가 공통의 결과로 계층화하여 본다면!)
    
- 또 다른 예로, 그림 6.8에서는 그림 6.7의 도표에 C가 추가되었다.
    - <img src="https://github.com/DoyoungKim12/causal-inference/blob/master/image_CH1/capture_18.PNG?raw=true">
    - C는 이뇨제로, 심장병 발병여부 L에 따른 결과이다. (병이 확진되면, 이뇨제 C를 투여)
    - C 역시 A와 Y의 공통 결과이기 때문에, C의 level별로 보았을 때 A와 Y는 연관되어있다.
    - 인과 그래프 이론은 이처럼 충돌점 L의 영향을 받는 C에 조건을 걸어도 상관관계의 흐름이 열리는 것을 보여준다.
    
- 여기서 다루지 않은 두 변수 간 상관관계의 원인은 무작위 다양성이다.
    - 실험 대상 집단의 수가 많으면 많을수록 줄어드는 chance association
    - 해당 문제에 대해서는 10장에서 다루기로 하고, 일단은 실험대상 집단의 수가 충분히 크다고 가정하자.
    
<br>

### 6.4 Positivity and consistency in causal diagrams
    
### 인과 도표에서의 양의 가능성과 일치성

- standardization과 IP weighting처럼 처리 효과를 정량화하는 방법들은 인과 도표 이론에서도 도출될 수 있다.
    - 해당 방식은 do-calculus라고 언급되는 경우가 많음.
    - 따라서 1장 5절에서 우리가 선택한 반사실적 이론은 실제로 하나의 특정한 접근방식이 아니라 단지 하나의 특정한 표기법을 특권화한 것이다.
    
- 그 표기법(notation)에 상관없이, 교체성, 양의 가능성, 일치성은 인과추론에서 요구되는 조건이다.
    - 교체성은 7장과 8장에서 다루고, 여기서는 양의 가능성과 일치성이 어떻게 그래프 언어로 번역되는지 다루겠다.
    
- 그래프 언어에서의 양의 가능성
    - 노드 L에서 처리 노드 A로 향하는 화살표가 비결정적(not deterministic)이라는 조건이다.
    
- 일치성의 첫번째 요소 : 잘 정의된 개입(well-defined interventions)
    - 처리 노드 A에서 결과 Y로 향하는 화살표가 (가설이기는 하지만) 비교적 모호하지 않은 개입이라는 것이다. 
    
- 이 책에서 논할 인과도표에서는, 양의 가능성은 암시적이며 일치성은 그 표기법 안에 내재되어있다.
    - 왜냐하면, 우리는 상대적으로 잘 정의된 개입인 처리 노드만을 고려할 것이기 때문이다.
    - 양의 가능성은 처리 노드로 들어오는 화살표들에 관한 것이고, 잘 정의된 개입은 처리 노드를 떠나는 화살표들에 대한 것이다.
    - 따라서, 처리 노드는 모든 다른 노드와 비교하여 다른 상태를 부여받는다.
        - 몇 저자들은 이 부분을 보다 명료히 하기 위해, '결정 노드'를 인과 도표에 추가하기도 한다.
        - 그러나 우리는 결정 노드까지 사용하진 않을 것이다. (우리는 언제나 A에 대한 잠재적 개입에 대해 명시적이기 때문...)
        - 다음 챕터에서 다룰 SWIGs(반사실적 변수를 나타내는 노드를 포함하는 도표)에서는 처리 노드에 다른 상태를 부여할 것이다.
        
- 아무튼 중요한 것은 무엇인가?
    - 충분히 잘 정의되지 않은 다층위의 처리 변수 노드는 인과 도표에 그려질 수 없다!
    - 이하 compound treatment에 대한 예시는 생략.
    
    
<br>

### 6.5 A structural classification of bias
    
### 편향의 구조적 분류

- 여기서는 샘플 사이즈의 문제가 아닌 구조적 문제에 따른 편향(systematic bias)에 대해 다룰 것이다.
    - 우리가 지금까지 무한한 샘플 수를 가정해왔기에 당연하다.
    
- 구조적 편항이란, 쉽게 말해 인과효과로 인해 발생한 것이 '아닌' 구조적 연관성이다. 
    - 인과도표는 연관성의 다른 원인들을 표현하기에 적합하므로, 그러한 구조적 편향들을 그 원인에 따라 나누어 살펴보기에 적합하다. 

- 지난 챕터에서 다루었던 편향의 치명적인 요소 : 교체성이 성립하지 않을 때
    - 전체 집단의 평균인과효과에서, 우리는 (무조건적) 교체성이 성립하지 않을 때 편항이 존재한다고 말한다.
    
- 교체성이 성립하지 않을 때는, 귀무가설(인과효과 없음)이 성립하더라도 편향이 발생한다.
    - 다시 말해, 처리가 결과에 대해 인과효과를 가지지 않더라도, 데이터 안에서 처리와 결과가 서로 상관관계를 가질 수 있다는 것이다.
    - 이런 경우, 우리는 교체성의 부재가 '귀무가설 하의 편향'(bias under the null)을 이끌었다고 표현한다.
    
- 위와 비슷한 맥락에서, 조건적 편향은 조건적 교체성이 특정 층위 하나에서라도 성립하지 않을 경우 존재한다고 말한다.

- 지금까지 교체성에 대해 수 차례 언급해왔지만, 그 교체성의 부재를 촉발하는 원인 구조에 대해 다루지는 못했다.
     - 이제 인과 도표를 활용하여, 2가지의 다른 원인 구조로 인해 교체성의 부재가 발생함을 설명할 것이다.
        1. 공통 원인(Common causes) : 처리와 결과가 같은 원인을 공유할 때, 상관관계와 인과관계의 측정 도구는 달라진다. 이러한 형태의 편향을 언급할 때, 보통 교란변수(confounding)라는 용어를 사용한다.
        2. 공통 원인에 대한 조건화 : 선택 편향(selection bias)에 대한 문제이다.
        
- 7장에서는 혼동변수 편향을, 8장에서는 선택 편향을 다룰 것이다.
    - 다시 한 번 말하지만, 둘 다 교체성의 부재로 인한 귀무가설 하의 편향이다.
    - 9장에서는 편향의 다른 원인을 다룬다 : 측정 오류(measurement error)
        - 지금까지 우리는, 모든 변수들이 완벽하게 측정되었다고 가정해왔다.
        - 그러나 현실에서는 어느 정도의 측정 오류가 예상된다. 
        
- 따라서, 다음 3개 장은 3가지의 구조적 편향에 초점을 맞출 것이다. (혼동변수, 선택, 측정)
    - 이러한 편향은 관찰연구와 무작위 실험 모두에서 발생할 수 있다.
    - 그 전에, 여기서 잠시 효과 수정의 존재 하에서의 인과도표에 대해 짚고 넘어가자.

<br>

### 6.6 The structure of effect modification
    
### 효과 수정의 구조

- 인과 도표는 효과 수정의 개념을 설명하는 데에는 그다지 도움이 되지 않는다.
    - 예를 들어보자. 심장수술이라는 처리 A와 사망이라는 결과 Y가 있을 때, A의 Y에 대한 인과효과를 밝히는 것은 어렵지 않다.
    - 그런데, 시술의 퀄리티를 나타내는 V를 고려하게 되었다고 해보자.
        - V의 수준에 따라, 인과효과는 차이가 있을 수 있다. (효과 수정 발생)
        - <img src="https://github.com/DoyoungKim12/causal-inference/blob/master/image_CH1/capture_20.PNG?raw=true">
        - V는 무작위로 할당되었다고 가정하기에, 위 도표에서처럼 A와는 독립이다.
        
- 중요한 2개의 주의사항이 있다.
    - 첫번째 : V가 A와 Y의 공통원인이 아니기 때문에, 위 도표는 타당하다고 볼 수 없다. 
        - 단지 인과적 질문이 V를 참조하기 때문에 도표에 포함되었을 뿐이다.
        - V와 Y를 매개하는, (V가 아닌) N이라는 변수로도 역시 효과수정을 정량화 할 수 있는 것이다.
    - 두번째 : 인과 도표는 효과 수정이 구체적으로 어떻게 일어났는지를 구분하지 않는다.
        - 구체적인 3가지 경우는 아래와 같다
            1. A의 Y에 대한 인과효과의 방향은 같고, 효과의 크기만 다른 경우 
            2. A의 Y에 대한 인과효과가 V의 수준에 따라 다른 방향을 갖는 경우 (질적 효과 수정)
            3. A의 Y에 대한 인과효과가 V의 수준에 따라 존재하지 않기도 하는 경우
        - 위의 도표로는 V에 의한 효과 수정이 어떤 경우에 해당하는지 알 수 없다.
        
- 위의 예시에서는 V가 결과에 대해 인과효과를 가졌지만, 실제로는 그렇지 않은 경우가 더 많다.
    - 오히려, 결과에 인과효과를 가지는 변수의 대리변수(surrogates)인 것이다.
    - <img src="https://github.com/DoyoungKim12/causal-inference/blob/master/image_CH1/capture_21.PNG?raw=true">
        - S가 V의 대리변수인 경우, S로 계층화한 분석결과에서 S의 수준에 따른 효과 수정이 관찰되겠지만, 실제 영향을 미친 변수는 V이다.
        - S는 대리 효과 수정자(surrogate effect modifier)이며, V는 인과 효과 수정자(causal effect modifier)이다.
        - 현실에서는 둘을 구분하기 어려우나, 어쨌든 효과 수정의 개념은 저 두가지로 구성되어 있다.
        - 이로 인한 혼동을 피하기 위해, 효과수정 대신 인과효과의 이질성이라는 표현을 쓰기도 한다.
        
- 대리 효과 수정자에서 관찰할 수 있는 상관관계는 때로는 공통원인(shared common causes)이나 그에 대한 조건(conditioning)을 씌우는 것에 기인하기도 한다.
    - <img src="https://github.com/DoyoungKim12/causal-inference/blob/master/image_CH1/capture_22.PNG?raw=true">
        - 위처럼 U가 V와 P의 공통원인일 경우, P는 대리 효과 수정자로 기능한다. (인과효과 수정자 V와 같은 원인을 공유하기 때문)
    - <img src="https://github.com/DoyoungKim12/causal-inference/blob/master/image_CH1/capture_23.PNG?raw=true">
        - 위처럼 S의 수준으로 나누어 관찰할 때, W가 S에 영향을 미친다면 W는 Y와 상관관계를 가질 수 있다. (W가 대리효과 수정자로 기능)
        - 예를 들어, S가 수술비이고 W가 생수 사용량이라고 한다면 생수 사용량은 Y와 인과관계가 없겠지만 비용에는 영향을 미치므로 S=0인 그룹에서는 Y와 상관관계를 가지게 된다.
        
<br><br><br><br>

## Chapter 7. CONFOUNDING

## 교란

- 다시 '누군가 하늘을 올려다볼 때, 다른 사람도 올려다볼까?'의 관찰 연구로 돌아가보자.
    - 첫 보행자가 하늘을 올려다보면, 두번째 보행자도 올려다보는 것을 발견했다고 치자.
    - 그러나, 하늘에서 천둥 소리가 날때도 하늘을 올려다보는 것을 발견했다면, 두번째 보행자의 올려다보는 행위는 무엇에 의한 것인지 분명치 않다.
    - 따라서, 누군가의 올려다보는 행위는 천둥소리의 존재에 의해 '교란'되었다고 볼 수 있다.
    
- 우리는 이러한 교란을, 단순히 처리집단과 비처리 집단간 교체성의 부재로 이해할 수 있다.
    - 교란 문제는 실험대상의 수가 충분히 많더라도 발생하는 문제이다.
    - 이 챕터에서는 교란의 정의와 그 조정 방법에 대해 알아볼 것이다.
    
<br>

### 7.1 The structure of confounding
    
### 교란의 구조

- 처리와 결과가 같은 원인을 가져 발생하는 편향인 교란의 구조는 인과 도표를 사용하여 나타낼 수 있다.
    - <img src="https://github.com/DoyoungKim12/causal-inference/blob/master/image_CH1/capture_24.PNG?raw=true">
    - 위 도표는 처리 A와 결과 Y, 그리고 그들의 공통원인인 L을 나타낸다.
    - 여기서 우리는 상관관계가 발생하는 2개 원인을 관찰할 수 있다.
        1. A에서 Y로 향하는 화살표
        2. L에서 각각 A와 Y로 향하는 화살표 : 이는 backdoor path의 예시이다.
    - L이 존재하지 않았다면, A와 Y사이의 상관관계는 그 인과관계로 인한 것이라고 말할 수 있다.
        - 그러나, L의 존재가 추가적인 상관관계 원인을 제공하여, 교란을 가져왔다.
        - 이로 인해, 상관관계를 인과관계로 해석할 수 없게 되었다.
        
- 관찰 연구에서 교란의 사례는 무수히 많다.
    - 사례 1 : 직업 관련 요소 (healthy worker bias)
        - 소방관으로 일하는 것 A와 사망률 Y는 공통원인 L(건강한 신체)을 가지고 있다.
        - L은 A와 양의 상관관계, Y와 음의 상관관계를 가지고, 이러한 편향은 healthy worker bias로 불린다.
    - 사례 2 : 임상적 판단 (channeling)
        - <img src="https://github.com/DoyoungKim12/causal-inference/blob/master/image_CH1/capture_25.PNG?raw=true">
        - 특정 질병 Y에 대한 약물 A의 효과는, 해당 약물이 특정 상태(L)인 사람에게 처방되었을 가능성이 높다면 교란될 것이다.
        - L은 Y의 직접적 원인일 수도, 또는 위 도표처럼 측정되지 않은 원인(U)을 공유할 수도 있다.
        - 이런 형태의 편향은 보통 channeling으로 불린다.
    - 사례 3 : 생활양식 (reverse causation)
        - <img src="https://github.com/DoyoungKim12/causal-inference/blob/master/image_CH1/capture_26.PNG?raw=true">
        - 행동 A가 결과 Y에 미치는 영향은 A가 다른 행동 L과 연관되어있을 때 교란된다.
        - L은 Y에 대해 인과효과를 가지며, A와 같이 발생하는 경향이 있다.
        - 이는 위의 도표에 나타나 있다. 측정되지 않은 변수 U가 개인의 성향이라고 한다면, 해당 변수는 2개 행동에 영향을 미칠 것이다.
        - 이런 형태의 교란은 L이 식별되지 않았을 때 reverse causation으로 불린다.
    - 사례 4 : 유전적 요인
        - 특정 DNA 배열 A가 Y라는 특성 발현에 미치는 영향은, Y와 인과관계를 가지면서 A와도 양의 상관관계를 가진 다른 DNA 배열 L에 의해 교란될 수 있다.
        - 3번 사례와 같은 구조로, linkage disequilibrium 또는 population stratification로 불린다.
    - 사례 5 : 사회적 요인
        - 65세 당시의 소득이 75세 당시의 장애에 미치는 영향은 55세 당시의 장애 여부에 의해 교란될 수 있다.
        - 도표 7.1의 구조와 같다.
    - 사례 6 : 환경 노출
        - 도표 7.3의 구조와 같고, 사례가 직관적으로 이해하기 어려워 생략함.
        
- 위의 모든 경우에서, 편향은 같은 구조를 가진다.
    - 처리와 결과가 함께 공유하는 원인(L 또는 U)이 있고, 그것이 A와 Y 사이의 배후 통로(backdoor path)를 여는 결과를 가져왔다.
    - 이처럼 공통 원인에 의한 편향을 교란(confounding)이라 하고, 그 외 다른 구조적 이유로 인한 편향은 다른 용어를 사용하자.
    
<br>

### 7.2 Confounding and exchangeability
    
### 교란과 교체성

- 이제 교란의 개념을 교체성의 개념과 연결시켜보자.
    - 정의를 명료히 하기 위해, 양의 가능성과 일치성은 만족되었으며 모든 변수들은 완벽하게 측정된 것으로 가정하자.
    
- 조건부 교체성이 만족될 경우, 계층화나 IP weighting 등의 조정을 통해 인과효과를 식별할 수 있다.
    - 그러나, 우리가 교란을 의심하게 될 경우, 중요한 문제가 발생한다.
    - "조건부 교체성을 만족하기 위해 측정된 공변량 L이 존재하는가?"
    - 이 문제에 답하기는 쉽지 않은데, 그 이유는 조건부 교체성을 고려하기에는 복잡한 인과 시스템이 직관적이지 못하기 때문이다.
    
- 이 챕터에서, 우리는 이 데이터를 생성한 인과 DAG를 알기만 하면 위 질문에 답할 수 있음을 보일 것이다.
    - 이를 위해, 일단 우리가 진짜 인과 DAG를 알고 있다고 가정하자. (어떻게 알게 되었는지는 중요하지 않다.)
    - 인과 DAG가 어떻게 조건부 교체성을 만족하기 위한 L이 있는지 알려준다는 것일까?
        - 2가지의 주된 접근방식이 존재한다. 
        - 배후 기준(the backdoor criterion)이 인과 DAG에 적용되거나, 인과 DAG를 SWIG로 전환하거나!
        - SWIG가 보다 직접적인 접근이지만, 배경 지식이 더 필요하므로 일단은 배후 기준에 대해 알아보자. (SWIG는 7.5에서 설명)
 
- 공변량 L의 집합이 배후 기준을 만족하는 조건은 2가지이다.
    1. A와 Y 사이의 모든 배후 경로가 L에 대한 조건 부여(conditioning)로 막혀있다.
    2. L은 처리 A의 자손(A를 부모 변수로 가지는 변수)을 포함하지 않는다.
    
- 조건부 교체성은 위와 같은 배후 기준을 만족하는 경우 성립하며, 이에 대한 간단한 증명은 SWIG 기반으로 진행될 예정이다.
    - 어쨌든, 우리는 이제 주어진 공변량 L의 집합에 대해 조건부 교체성의 성립 여부를 알 수 있다.
    - 따라서, 처리 A의 자손이 아닌 변수들의 부분집합을 모두 고려하여, 조건부 교체성 성립 여부에 답할 수 있다.
        - (사실, 이에 답하기 위해 고려해야 하는 부분집합의 수를 극적으로 줄여주는 알고리즘이 있긴 하다.)
        
- 이제 배후 기준을 교란과 연결시켜보자. 
    - 배후 기준이 만족되는 두 조건은 아래와 같다.
        1. 처리와 결과 사이에 공통 원인이 없어야 한다. 
        2. 처리와 결과 사이에 공통 원인이 있지만, 모든 배후경로를 차단하기에 충분한 부분집합이다.
            - 처리와 결과 사이에 공통 원이이 있으면 이를 교란으로 보지만, 잠재교란변수(residual confounding)는 없는 상태
            - 보다 간략하게 표현하자면, 측정되지 않은 교란이 없는 상태! (잠재교란변수 = 측정되지 않은 교란)
    - 첫번째 조건은 처리가 무작위로 적용되기 때문에 교란이 예상되지 않는 무작위 실험을 뜻한다.
        - 처리가 조건 없이 무작위로 적용된다면, 공통 원인이 존재하지 않기 때문에(= 열린 배후 경로가 없기 때문에)실험군과 대조군은 교체성을 가진다.
        - 조건 없는 교체성은 처리와 결과 사이에 공통 원인이 없다는 것과 같다.
        
    - 두번째 조건은 조건부 무작위 실험을 뜻한다.
        - 특정 변수들에 대해 같은 값을 가진 개인들에 한하여 처리가 적용될 확률이 같다는 것을 의미한다.
        - 물론, 변수의 값에 따라 처리가 적용될 확률은 달라진다.
        - 이러한 실험 디자인은 교란이 보장되는데, 
            - 첫번째로, 그 변수가 결과의 위험요소일 때.
            - 두번째로, 그 변수가 결과의 원인이 되거나 측정되지 않은 결과의 원인의 자손인 경우이다. (각각 그림 7.1과 7.2의 경우)
            - 이런 경우가 바로 열린 배후 경로가 있는 상황이다.
        - 그러나, 그러한 변수에 조건을 설정하는 것으로 배후 경로를 차단할 수 있고, 조건부 교체성이 성립할 수 있다.
            - 이처럼 조건이 설정되었다면, 우리는 (처리의 자손이 아닌) 측정된 교란변수 집합을 "교란 조정에 충분한 집합"이라 부른다.
            - 이 말은, 측정된 교란변수의 수준별로 교체성이 성립하게 된다는 뜻이다.
            
- 심장 이식 예제로 다시 돌아가보자.
    - 심장질환 유무(L)는 심장이식(A)과 사망(Y)의 공통 원인이 된다.
    - L에 따른 조건부 교체성만이 성립 가능하다.

- 배후 기준은 교란의 방향과 강도에 대해서는 답하지 않는다.
    - 여러 교란이 얽혀 그 강도와 뱡향을 특정하기 어려운 상황이 발생할 수 있다.

<br>

### 7.3 Confounding and the backdoor criterion
    
### 교란과 배후조건

- 우리는 이제 Y에 대한 A의 인과효과가 식별가능한지를 판별하기 위한 배후조건의 적용 예시 몇가지를 살펴볼 것이다.
    - 그리고, (인과효과의 식별이) 가능하게 된다면 조건부 교체성을 보장하기 위해 어떤 변수들이 요구되는지도 역시 살펴볼 것이다.
    - 이 챕터의 모든 인과 DAG는 조건이 없는 완벽하게 측정된 노드를 포함한다는 점을 기억하자.
    
- 그림 7.1을 보면, A와 Y가 공통원인 L을 공유하고 있기 때문에 교란이 존재한다.
    - 다시 말해, A와 Y 사이에 L을 통한 열린 배후 경로가 존재한다.
    - 하지만 이 배후경로는 L에 조건을 부여하여 차단될 수 있다.
    - 따라서, 모든 개인에 대해 L의 데이터를 모으게 된다면 측정되지 않은 L에 의한 교란은 없어지게 되는 셈이다.
    
- 그림 7.2를 보면, A와 Y가 (측정되지 않은) 공통원인 U을 공유하고 있기 때문에 교란이 존재한다.
    - 이런 배후경로는 이론적으로 막혀있다고 본다.
    - 그리고 L에 조건을 부여하여 배후경로를 막는 방법도 있다.
    - 따라서, 여기서도 측정되지 않은 L에 의한 교란은 없다.
    
- 그림 7.3을 보면, 




