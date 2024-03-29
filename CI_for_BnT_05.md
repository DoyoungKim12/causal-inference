<br><br><br><br>

# 5. The Unreasonable Effectiveness of Linear Regression

<br><br>

## All You Need is Regression
- 인과추론을 다룰 때, 우리는 각 개인에 대해 어떻게 두 가지의 잠재적 결과가 있는지 살펴보았다.
  - <img src="https://render.githubusercontent.com/render/math?math=Y_0"> : 개인이 처리를 못받았을 때의 효과
  - <img src="https://render.githubusercontent.com/render/math?math=Y_1"> : 개인이 처리를 받았을 때의 효과
  - 처리 <img src="https://render.githubusercontent.com/render/math?math=T">를 0 또는 1로 설정하는 것은 이 잠재적 결과들 중 하나가 실현되게 하고, 나머지 하나는 영영 알아낼 수 없도록 만든다. 이것이 개인의 처리효과인 <img src="https://render.githubusercontent.com/render/math?math=\tau_i = Y_{1i} - Y_{0i}">를 알 수 없게 하는 것이다. (너무 당연한 이야기)
  - <img src="https://render.githubusercontent.com/render/math?math=Y_i = Y_{0i}(1-T_i) %2B T_iY_{1i}"> (T가 1이면 Y1만 남고, T가 0이면 Y0만 남는 형태)
<br><br>

- 따라서, 현재로써는 평균 인과 효과를 추정하는 단순한 문제에만 집중해보자. 우리는 다른 사람에 비해 특정 처리에 더 잘 반응하는 사람들이 있다는 사실을 알지만, 역시 그들이 누구인지 알 수도 없다. (그걸 알아내는) 대신, 우리는 그 처리가 **평균적인** 효과가 있는지 알아내볼 것이다. 
  - <img src="https://render.githubusercontent.com/render/math?math=ATE = E[Y_1 - Y_0]">  
  - 이렇게 하면 처리효과가 상수로 일정한 단순한 모델을 얻을 수 있다. (<img src="https://render.githubusercontent.com/render/math?math=Y_{1i} = Y_{0i} %2B \kappa">) 만약 <img src="https://render.githubusercontent.com/render/math?math=\kappa"> 가 양수라면, 우리는 그 처리가 평균적으로 양의 효과를 가진다고 말할 수 있다. 특정 사람들에게 안좋게 작용할지라도, 평균적으로 그 영향은 양의 효과일 것이다. 
<br><br>

- 또한 우리가 ATE를 (편향 때문에) 단순히 처리군과 대조군 각각의 평균 차이로 구할 수 없었다는 사실도 기억해내보자. 편향은 주로 처리군과 대조군이 처리효과 그 자체가 아닌 다른 이유로 인해 다를 때 발생한다. 이걸 살펴보는 방법은 그들이 잠재적 결과인 <img src="https://render.githubusercontent.com/render/math?math=Y_0">에서 얼마나 다른지를 추정해보는 것이다. 
  - <img src="https://github.com/DoyoungKim12/causal-inference/blob/master/img_BnT/bnt_17.PNG?raw=true">
<br><br>

- 이전에 우리는 무작위 실험, 또는 무작위 비교실험(Randomised Controlled Trial, RCT)으로 어떻게 편향을 제거하는지 살펴보았다. RCT는 처리군과 대조군을 동질하게 만들고 그것이 편향이 사라지는 이유가 되었다. 우리는 또한 우리의 추정치에 대한 불확실성의 정도를 어떻게 설정해야 할지도 배웠다.
- 예시 (온라인 강의 vs. 대면강의)
  - T=0 : 대면강의, T=1 : 온라인 강의 일 때, 학생들은 두 타입의 강의 중 하나에 무작위로 배정되어 그들의 성과를 시험으로 평가한다. 우리는 두 그룹을 비교할 수 있는 A/B 테스트 함수를 설계하여 평균 처리효과와 신뢰구간까지 제공해보았다.  
<br><br>

- 이제, 이 모든 것들을 인과추론의 참된 일꾼, **선형 회귀**로 해결할 수 있다는 것을 배워볼 시간이다! 
- 이렇게 생각해보자. 처리군과 대조군을 비교하는 것이 디저트로 먹는 사과라면, 선형회귀는 차갑고 크리미한 티라미수이다. (뭔소리? 암튼 선형회귀가 좋다는 거겠지)
<img src="https://github.com/DoyoungKim12/causal-inference/blob/master/img_BnT/bnt_18.png?raw=true">
<br><br>

- 이 아름다움이 어떻게 작동하는지 살펴보자. 아래의 코드에서, 우리는 온라인 강의와 대면 가의를 비교하는 정확히 같은 분석을 하려고 한다. 하지만 신뢰 구간 등의 수학 어쩌고를 하는 대신, 우리는 그냥 회귀를 돌릴 것이다. 보다 구체적으로, 우리는 아래의 모델을 추정할 것이다.
- <img src="https://render.githubusercontent.com/render/math?math={exam}_{i} = \beta_0 %2B \kappa {Online}_i %2B \mu_i"> 
  
  - Online : 처리를 표시하는 것으로, 더미 변수임 (1 또는 0의 값만 가지고, 이 경우에는 온라인 수업일 때 1, 대면 수업일 때 0)
  
  - <img src="https://render.githubusercontent.com/render/math?math=E[Y|T=0] = \beta_0"> 이고, <img src="https://render.githubusercontent.com/render/math?math=E[Y|T=1] = \beta_0 %2B \kappa"> 이니까 <img src="https://render.githubusercontent.com/render/math?math=\kappa">가 우리가 찾고자 하는 ATE가 되겠다.

```python
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import graphviz as gr
%matplotlib inline

data = pd.read_csv("data/online_classroom.csv").query("format_blended==0")
result = smf.ols('falsexam ~ format_ol', data=data).fit()
result.summary().tables[1]
```
<img src="https://github.com/DoyoungKim12/causal-inference/blob/master/img_BnT/bnt_19.PNG?raw=true">
(위 코드 실행의 결과)
<br><br>

- 정말 놀랍다. 우리는 ATE를 추정할 수 있을 뿐만 아니라, 공짜로 신뢰구간과 P-Value까지 확인할 수 있다! 게다가, 우리는 회귀식이 정확히 우리가 하고자 했던 것을 수행하고 있는 것을 볼 수 있다.
  - T=0, 즉 대면수업을 받은 샘플들의 표본평균 <img src="https://render.githubusercontent.com/render/math?math=E[Y|T=0]">은 intercept의 값과 정확히 일치한다.
  - 온라인 포맷(format_ol)의 coef(계수)는 처리군과 대조군의 차이와 정확히 일치한다. (<img src="https://render.githubusercontent.com/render/math?math=E[Y|T=1] - E[Y|T=0]">)

<br><br>

## Regression Theory
- 선형회귀가 어떻게 구성되고 (계수를) 어떻게 추정하는지와 같은 깊은 이야기까지 꺼내진 않을 것이다. 그러나, 약간의 이론은 큰 도움이 될 것이다.
- 먼저, 회귀는 이론상 최적의 선형 예측 문제를 푼다.
  - <img src="https://github.com/DoyoungKim12/causal-inference/blob/master/img_BnT/bnt_20.PNG?raw=true">
  - <img src="https://render.githubusercontent.com/render/math?math=\beta^*">을 파라미터들의 벡터라고 해보자. <br>선형회귀는 오차 제곱의 평균, 즉 mean squared error(MSE)를 최소로 하는 파라미터의 값을 찾는 것이다. <br>만약 이걸 미분하여 0이 되도록 한다면, 이 문제에 대한 선형 해는 아래와 같이 주어진다.
  - <img src="https://render.githubusercontent.com/render/math?math=\beta^* = E[X_i^'X_i]^{-1}E[X_i^'Y_i]">
  - 이 베타를 추정하는 방법은 아래와 같다. 하지만 내 말을 그대로 믿진 말라. 수식보다 코드를 더 잘 이해한다면, 아래 코드를 보아라.
  - <img src="https://render.githubusercontent.com/render/math?math=\hat\beta = (X_i^'X_i)^{-1}X_i^'Y_i">

```python
X = data[["format_ol"]].assign(intercep=1)
y = data["falsexam"]

def regress(y, X): 
    return np.linalg.inv(X.T.dot(X)).dot(X.T.dot(y))

beta = regress(y, X)
beta
```

- 인과추론에서는 주로 변수 T가 결과 y에 미치는 인과적 영향만을 추정하고 싶어하기 때문에, 우리는 이 효과를 추정하기 위해 하나의 변수만을 고려하는 회귀식을 사용할 것이다. 만약 다른 변수를 추가하게 되더라도, 그건 보통 보조에 불과하다. 다른 변수를 추가하는 것이 처리의 인과효과를 추정하는 데에 도움을 줄 수는 있겠지만, 우리는 그들의 파라미터(계수)를 추정하는 것에 별로 관심이 없다. 
- 단변량 회귀의 변수 T에 대해, 이와 연관된 파라미터는 아래와 같이 주어진다.
  - <img src="https://render.githubusercontent.com/render/math?math=\beta_1 = \frac{Cov(Y_i,T_i)}{Var(T_i)}">
  - T가 무작위 할당되었다면, <img src="https://render.githubusercontent.com/render/math?math=\beta_1">은 ATE이다. 
<br><br>

- 다른 모든 변수가 보조이고, 우리가 정말로 궁금한 것은 T와 연관된 파라미터 <img src="https://render.githubusercontent.com/render/math?math=\kappa">라고 하자. <img src="https://render.githubusercontent.com/render/math?math=\kappa">는 아래의 수식을 통해 계산할 수 있다. 
  - <img src="https://render.githubusercontent.com/render/math?math=y_i = \beta_0 %2B  \kappa T_i %2B \beta_1X_{1i} %2B ... %2B \beta_kX_{ki} + \mu_i">
  - <img src="https://render.githubusercontent.com/render/math?math=\kappa = \frac{Cov(Y_i,\tilde{T_i})}{Var(\tilde{T_i})}">
    - <img src="https://render.githubusercontent.com/render/math?math=\tilde{T_i}">는 T에 대한 X1 ~ Xk의 모든 공변량의 회귀식으로부터 나온 잔차(residual)이다. 이제 이게 얼마나 멋진 것인지 감사하자. (??뭐지) 이것은 다변량 회귀의 계수가 **모델의 다른 변수들의 효과를 고려하고 난** 동일한 회귀의 이변량 계수임을 의미한다. 인과추론의 관점에서, <img src="https://render.githubusercontent.com/render/math?math=\kappa">는 모든 다른 변수들을 예측에 사용해버린 후의 T의 이변량 계수이다. 
<br><br>

- 이 뒤에는 멋진 직관이 숨겨져 있다. 우리가 다른 변수들을 사용해서 T를 예측할 수 있다면, 이는 곧 무작위가 아니라는 것을 의미한다. 하지만, 일단 우리가 다른 사용 가능한 변수들을 제어한다면 T가 무작위만큼 좋도록 만들 수 있다. 이렇게 하기 위해, 우리는 다른 변수들로부터 T를 예측하기 위해 선형회귀를 사용한다. 그리고 그 회귀식의 잔차인 <img src="https://render.githubusercontent.com/render/math?math=\tilde{T_i}">를 가져온다. 당연히, <img src="https://render.githubusercontent.com/render/math?math=\tilde{T_i}">는 우리가 이미 T를 예측할 때 썼던 다른 변수들에 의해 예측되지 못한다. 아주 우아하게 말해보자면, <img src="https://render.githubusercontent.com/render/math?math=\tilde{T_i}">는 그 어떤 변수와도 연관되지 않은 처리이다. 
<br><br>

- 참고로, 이 또한 선형 회귀의 속성이다. 잔차는 잔차를 생성한 모형의 변수와 항상 직교하거나 상관 관계가 없다. 더 멋진 사실은 이것들이 수학적 진실이라 데이터가 어떻게 생겨먹든 상관이 없다는 것이다. 
```python
e = y - X.dot(beta)
print("Orthogonality imply that the dot product is zero:", np.dot(e, X))
X[["format_ol"]].assign(e=e).corr()

# Orthogonality imply that the dot product is zero: [7.81597009e-13 4.63984406e-12]
```
<br><br>

## Regression For Non-Random Data
- 여기까지 우리는 무작위 실험의 데이터를 다뤄왔다. 그러나, 알다시피 그런 실험은 매우 비용이 많이 들거나 말그대로 불가능하다. 이러한 이유로 인해, 이제부터는 무작위가 아니거나 관찰된 데이터를 살펴볼 것이다. 아래의 예시에서 우리는 추가 교육을 받은 기간이 시급에 미치는 영향을 추정해볼 것이다. 이미 짐작했겠지만, 교육에 대해 실험을 수행하는 것은 매우 어렵다. 이런 경우에 우리가 가진 것은 관찰 데이터가 전부이다.
- 먼저, 아주 쉬운 모델을 가정해보자. 우리는 시급에 로그를 씌운 값을 교육 기간으로 회귀할 것이다. 로그글 씌운 것은 파라미터 추정치가 퍼센트(%)로의 의미를 갖게 하기 위함이다. 즉, 우리는 1년의 추가 교육이 시급 인상의 몇 퍼센트 기여했는지 말할 수 있다. 
  - <img src="https://render.githubusercontent.com/render/math?math=log(hwage)_i = \beta_0 %2B \beta_1{educ}_i %2B \mu_i">

```python
wage = pd.read_csv("./data/wage.csv").dropna()
model_1 = smf.ols('np.log(hwage) ~ educ', data=wage.assign(hwage=wage["wage"]/wage["hours"])).fit()
model_1.summary().tables[1]
```
<img src="https://github.com/DoyoungKim12/causal-inference/blob/master/img_BnT/bnt_21.PNG?raw=true">

- <img src="https://render.githubusercontent.com/render/math?math=\beta_1">의 추정치는 0.0536으로, (0.039, 0.068)의 95% 신뢰구간을 가진다. 이는 모델이 교육받는 년수가 1년 증가할 때마다 임금이 5.3%씩 증가할 것으로 예측했다는 것을 의미한다. 
- 당연히, 우리가 이 단순한 모델로 예측할 수 있는 것이 정확한 것은 아니다. 예측이 곧 인과성을 의미하는 것이 아니다. 사실 이 모델은 편향되어 있다고 의심해볼만 한것이, 이 데이터는 무작위 실험으로부터 나온 것이 아니다. 부모의 수입이나 교육수준 같은 것들의 영향으로 교육수준이 정해지는 거라면, 오히려 교육을 받지 않는 것이 임금 수준에 더 도움이 된다고 주장할 수도 있다.
- 다행히도, 우리의 데이터로 다른 많은 변수들에 접근할 수 있다. 우리는 부모의 교육수준, IQ, 근속기간, 기혼여부, 인종 등의 변수를 살펴볼 수 있다. 이 모든 변수들을 모델에 포함시켜 계수를 추정할 수 있고, 수식은 아래와 같다. 
  - <img src="https://render.githubusercontent.com/render/math?math=log(hwage)_i = \beta_0 %2B \kappa {educ}_i %2B \beta X_i %2B \mu_i">
- 이 방법이 어떻게 편향의 문제를 해결하는지 이해하기 위해, 다변량 회귀의 이변량 분석을 다시 살펴보자. 해당 수식은 아래와 같다.
  - <img src="https://render.githubusercontent.com/render/math?math=\kappa = \frac{Cov(Y_i,\tilde{T_i})}{Var(\tilde{T_i})}">
  - 이 수식은 다른 모든 변수(부모 교육수준, IQ 등등)를 사용하여 educ를 예측할 수 있다고 말하고 있다. 우리가 이걸 하고나면, 이전에 포함되었던 모든 변수들과 무관한 버전의 educ를 얻을 수 있다. (이제 교육 수준이 높은 것은 IQ 때문이고, 따라서 교육수준이 높은 임금의 원인은 아니다 라는 주장은 못할 것이다. 우리가 IQ를 모델에 포함시킨다면, IQ의 수준을 고정한 채로 추가적인 교육 기간의 효과를 관찰할 수 있게 된다.) 지금까지 본 것들이 무엇을 의미하는지 이해하기 위해 잠시 멈춰 생각해보자. 만약 우리가 처리군과 대조군의 모든 요소들을 동질하게 하는 무작위 통제 실험을 하지 못하더라도, 회귀는 그 요소들을 모델에 포함하는 것만으로 두 그룹의 동질성을 확보할 수 있다. 데이터가 랜덤이 아니라도 말이다!

```python
controls = ['IQ', 'exper', 'tenure', 'age', 'married', 'black',
            'south', 'urban', 'sibs', 'brthord', 'meduc', 'feduc']

X = wage[controls].assign(intercep=1)
t = wage["educ"]
y = wage["lhwage"]

beta_aux = regress(t, X)
t_tilde = t - X.dot(beta_aux)

kappa = t_tilde.cov(y) / t_tilde.var()
kappa

# kappa 값은 0.041147191010057635
```

- 우리가 방금 추정한 이 계수는 같은 IQ, 같은 경력, 같은 나이를 가진 사람들이 1년의 추가 교육으로 시급의 4.11% 상승을 기대할 수 있다고 말해주고 있다. 이는 최초의 교육수준만을 변수로 사용한 모델이 편향되었다는 우리의 의심을 확인시켜준다. (그 모델은 교육의 영향을 과대평가했다.) 그리고 이렇게 직접 코딩을 하는 대신 남들이 짠 코드를 사용한다면, 우리는 이 추정치의 신뢰 구간도 쉽게 구할 수 있다. 
```python
model_2 = smf.ols('lhwage ~ educ +' + '+'.join(controls), data=wage).fit()
model_2.summary().tables[1]
```
<br><br>

## Omitted Variable or Confounding Bias
- 남은 문제는 우리가 추정한 이 계수가 인과관계를 의미하냐는 질문에 대답하는 것이다. 불행하게도, 확실히 그렇다고 대답할 수는 없다. 첫번째로 다룬 단순한 모델의 경우에서는 아마도 인과관계를 의미한다고 말하지 못할 것이다. 그 모델에서는 교육수준과 임금 양쪽 모두와 연관된 중요한 변수들을 빠뜨렸다. 그 변수들을 통제하지 않는다면, 추정된 효과는 모델에 포함되지 않은 다른 변수들의 효과까지 같이 포착하게 된다. 이러한 편향이 어떻게 작용하는지 더 잘 이해하기 위해, 교육이 임금에 미치는 영향에 대한 진짜 모델(당연히 현실에는 존재하지 못할...)이 다음과 같이 있다고 하자.
  - <img src="https://render.githubusercontent.com/render/math?math={Wage}_i = \alpha %2B \kappa{Educ}_i %2B A^`_i\beta %2B \mu_i">
<br><br>

- 임금은 <img src="https://render.githubusercontent.com/render/math?math=\kappa">의 크기로 측정되는 교육과 (벡터 A로 표기된) 추가적인 능력 요소의 영향을 받는다. 만약 우리가 능력 변수를 우리 모델에서 누락한다면, 우리의 추정치 <img src="https://render.githubusercontent.com/render/math?math=\kappa">는 아래와 같을 것이다.
  - <img src="https://github.com/DoyoungKim12/causal-inference/blob/master/img_BnT/bnt_22.PNG?raw=true">
  - <img src="https://render.githubusercontent.com/render/math?math={\delta}_A">는 Educ에 대한 A의 회귀식으로부터 나온 계수들의 벡터를 의미한다. 
  - 여기에서 중요한 점은 추정치가 우리가 원했던 정확한 <img src="https://render.githubusercontent.com/render/math?math=\kappa">가 아닐 것이라는 점이다. 그 대신, 짜증나는 추가항(<img src="https://render.githubusercontent.com/render/math?math={\beta}^`{\delta}_A">)이 딸려온다. 이 추가항은 누락된 변수 A가 임금에 미치는 영향을 의미한다. 이것은 조슈아 앵그리스트라는 사람이 (학생들이 그것을 명상하면서 암송할 수 있도록 하기 위해) 그것을 만트라(기도문)로 만들었다는 점에서 경제학자들에게 중요하다. (???기도문??)
  - "Short equals long <br> plus the effect of omitted <br> times the regression of omitted on included"
  - 여기서 the short regression은 특정 변수를 누락한 것이고, the long은 그 변수를 포함한 것이다. 이 공식 또는 기도문은 편향의 속성에 대한 더 깊은 인사이트를 제공한다. 먼저, 누락된 변수가 종속변수 Y에 미치는 영향이 없다면, 편향 항은 0일 것이다. 두번째로, 누락된 변수가 처리변수 T에 미치는 영향이 없다면, 역시 편향 항은 0일 것이다. 이것 역시 직관적으로 말이 된다. 만약 교육에 영향을 미치는 모든 것들이 모델에 포함되었다면, 교육 효과의 추정치가 (임금에도 영향을 미치는) 다른 무언가에 대한 교육의 상관관계와 섞여있을 리 없다. 
<img src="https://github.com/DoyoungKim12/causal-inference/blob/master/img_BnT/bnt_23.png?raw=true">
<br><br>

- 좀더 간결하게 설명하자면, 모든 교란 변수가 모델에서 설명되었을 때 OVB(Omitted Variable Bias)가 없다고 말할 수 있다. 그리고 여기서 인과 그래프에 대한 지식을 활용해볼 수 있다. 교란변수는 처리와 결과 모두의 원인이 되는 변수이다. 이 임금 예시에서는 IQ가 교란변수이다. (높은 IQ를 가진 사람은 학습을 더 쉽게 하기 때문에 교육을 더 많이 받을 가능성이 높고, 더 생산성이 높기 때문에 더 높은 임금을 받을 수 있다.) 교란변수는 처리와 결과 모두에 영향을 주기 때문에, 우리는 그들을 T와 Y로 향하는 화살표와 함께 표시해두었다. 여기서 나는 그것들을 W라고 표기할 것이다. 그리고 양의 인과효과를 붉은색으로, 음의 인과효과를 파란색으로 표시했다.
<img src="https://github.com/DoyoungKim12/causal-inference/blob/master/img_BnT/bnt_24.PNG?raw=true">

- 양의 편향의 예시 : 첫번째 예시에서 IQ는 교육과 임금에 모두 영향을 미치는데, 우리가 IQ를 모델에서 고려하지 않으면 교육의 효과가 실제 영향보다 더 커보이게 된다.
- 음의 편향의 예시 : 도시 내 폭력 사건 발생 수와 경찰력의 관계에 대한 것이다. 우리가 볼 때, 경찰의 수가 많은 도시에서 더 많은 폭력 사건이 발생하는 것처럼 보인다. 그러나 범죄라는 변수가 폭력 사건 수와 경찰 수를 모두 증가시키는 원인이 되기 때문에, 실제로는 경찰 수가 폭력 사건을 줄이는 원인이 되더라도 그에 대한 고려가 없다면 (범죄의 영향때문에) 반대로 해석할 수 있다.   
<br><br>

- 인과 그래프는 회귀와 RCT 모두가 교란 편향을 어떻게 수정하는지를 보여줄 수 있다. RCT는 교란변수와 처리변수 사이의 연결을 차단한다. T가 랜덤이 되도록 하여 그 무엇도 T의 원인이 되지 못하도록 하는 것이다. 이와 달리, 회귀는 교란변수 W를 고정된 수준으로 둔 채로 T의 효과를 비교하여 해결한다. 그 수준이 고정되어있기 때문에, W는 T와 Y에 영향을 줄 수 없게 된다. 
- 자, 이제 다시 질문으로 돌아가서, 교육이 임금을 미치는 영향에 대해 추정한 계수가 인과관계를 보여주는 것일까? 미안하지만 그것은 모든 교란변수가 모델에 포함되었는지에 대한 우리의 논쟁 실력에 달렸다. 그리고 개인적으로는 제대로 포함되지 않았다고 본다. (예를 들어, 우리가 계속 보아온 임금 모델에서 부모의 재력은 포함되지 않았다. 교육 수준도 재력의 프록시일 뿐이다. 야망도 포함되어야 한다. 야망이 있는 자는 교육도 더 많이 받고, 더 높은 임금을 위해 노력할 것이다.) 이는 곧 무작위가 아니거나 관찰된 데이터에 대한 인과추론은 항상 약간의 의구심을 가지고 바라봐야 한다는 사실을 말해준다. 우리는 절대로 모든 교란변수가 고려되었다고 확신할 수 없다. 
<br><br>

## Key Ideas
- 회귀 분석을 사용하여 A/B테스트를 수행하고 신뢰 구간을 편리하게 찾는 방법을 확인했다.
- 회귀가 예측 문제를 푸는 방법과 그것이 CEF에 대한 최선의 선형 근사치라는 것을 공부했다.
- 이변량의 경우, 처리의 계수가 처리와 결과 사이의 공분산을 처리의 분산으로 나눈 것이다. 
- 다변량의 경우로 확장하여, 회귀 분석을 통해 처리 계수에 대한 해석을 부분적으로 수행하는 방법을 알아냈다. (포함된 다른 모든 변수들을 상수화하여 처리의 결과에 대한 효과를 해석)
- 누락된 변수의 회귀에 대한 효과를 확인하고, 그 원인이 무엇인지 살펴보았다. 
- 마지막으로, 인과 그래프를 사용하여 RCT와 회귀분석에서 교란변수를 수정하는 방법을 확인했다.  
<br><br>
