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
        - Standardization, IP weighting : 한계효과와 조건부효과 모두 측정가능
        - stratification/restriction, and matching : 특정 서브그룹의 조건부효과만 측정 가능
    - 네가지 방법 모두 교체성과 양의 가능성이 요구된다.
    
- 효과 수정이 없다면, 이 네가지 접근방법으로 계산된 효과 측정치는 모두 동일하다.





