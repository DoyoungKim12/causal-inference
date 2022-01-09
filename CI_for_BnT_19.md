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
