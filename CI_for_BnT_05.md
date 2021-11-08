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
  - <img src="https://render.githubusercontent.com/render/math?math=E[Y|T=0] = \beta_0"> 이고, <img src="https://render.githubusercontent.com/render/math?math=E[Y|T=1] = \beta_0 + \kappa"> 이니까 <img src="https://render.githubusercontent.com/render/math?math=\kappa">가 우리가 찾고자 하는 ATE가 되겠다.

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
<img src="https://github.com/DoyoungKim12/causal-inference/blob/master/img_BnT/bnt_19.png?raw=true">
(위 코드 실행의 결과)
<br><br>

- 정말 놀랍다. 우리는 ATE를 추정할 수 있을 뿐만 아니라, 공짜로 신뢰구간과 P-Value까지 확인할 수 있다! 게다가, 우리는 회귀식이 정확히 우리가 하고자 했던 것을 수행하고 있는 것을 볼 수 있다.
  - T=0, 즉 대면수업을 받은 샘플들의 표본평균 <img src="https://render.githubusercontent.com/render/math?math=E[Y|T=0]">은 intercepy의 값과 정확히 일치한다.
  - 온라인 포맷(format_ol)의 coef(계수)는 처리군과 대조군의 차이와 정확히 일치한다. (<img src="https://render.githubusercontent.com/render/math?math=E[Y|T=1] - E[Y|T=0]">)
<br><br>

## Regression Theory











