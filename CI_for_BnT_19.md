<br><br><br><br>

# 5. Evaluating Causal Models

<br><br>

## 인과 모델의 평가
- 인과관계에 대한 대부분의 자료에서, 연구자들은 그들의 방법이 옳았는지를 확인하기 위해 인위적으로 만들어진 데이터를 사용한다. 우리가 When Prediction Fails 챕터에서 했던 것과 같이, 연구자들은 각 개인의 처리 상태에 따른 2가지 결과를 모두 만들어낸다. 따라서 그들은 그들의 모델이 처리 효과를 올바르게 포착하는지 확인할 수 있다. 이는 학문적인 목적으론 나쁘지 않지만, 현실에서의 우리는 그런 사치를 누릴 수 없다. 이러한 기술을 실제 산업에 적용할 때, 우리는 왜 우리 모델이 더 나은지, 왜 그것이 현재의 모델을 대체해야 하는지, 왜 그것이 비참하게 실패하지 않는지를 증명하기 위해 계속해서 질문을 받을 것이다. 이것은 너무 중요해서, 왜 우리가 인과적 추론 모델을 실제 데이터로 어떻게 평가해야 하는지를 설명하는 어떠한 자료도 볼 수가 없는건지 이해가 되질 않는다.
<br>

- 그 결과로, 인과추론 모델을 적용하려는 데이터 과학자들은 경영진이 그들을 신뢰하도록 설득하는 일에 어려움을 겪는다. 그들이 취하는 접근법은 이론이 얼마나 적절한지, 모델을 학습시키는 동안 얼마나 세심하게 작업했는지를 보여주는 것이다. 불행하게도, train-test split 검증 패러다임이 표준이 된 세상에서 이는 쉽지 않을 것이다. 모델의 퀄리티는 아름다운 이론보다 뭔가 더 믿을만한 것에 기반한 것이어야 한다. 생각해보자. 머신러닝은 예측모델의 검증이 매우 직관적이라는 바로 그 점 때문에 큰 성공을 거둘 수 있었다. 예측이 실제 일어난 일과 일치한다는 것을 보면 뭔가 마음이 놓이는 게 있기 때문이다. 
<br>

- 안타깝게도, 인과추론의 경우에서 train-test 패러다임같은 뭔가를 이뤄낼 방법은 조금도 분명히 보이지 않는다. 이는 인과추론이 관찰되지 않은 값을 추정하는 것에 관심이 있기 때문이다(<img src="https://render.githubusercontent.com/render/math?math=\frac{\delta y}{\delta t}">). 우리가 볼 수 없다면, 우리의 모델이 이를 잘 추정한다는 것을 어떻게 알 수 있을까? 기억하자. 모든 개체는 처리와 결과의 관계를 나타내는 선의 기울기로 표시된 반응성(responsiveness)을 가지고 있지만 우리가 이를 측정할 수는 없다. 
<br>

<img src="https://github.com/DoyoungKim12/causal-inference/blob/master/img_BnT/bnt_2_1.PNG?raw=true">
<br>

- 이것은 우리의 머리를 감싸쥐게 만드는 매우 매우 어려운 일이고, 해답에 가까운 것을 찾는데 몇 년이 걸렸다. 확정적인 것은 아니지만 실제로 효과가 있고 설명력도 가지고 있는데, 머신러닝의 train-test 패러다임과 유사한 방식으로 인과적 추론에 접근했으면 한다. 비결은 집계된 탄력성을 사용하는 것이다. 탄력성을 개별로 추정할 수는 없더라도, 전체 그룹에 대해서는 할 수 있고 그것이 우리가 여기서 활용할 아이디어이다.
<br>

- 이 챕터에서는 무작위가 아닌 데이터셋을 사용하여 인과모델을 추정하고, 이를 무작위 데이터로 평가해볼 것이다. 또다시, 가격이 아이스크림의 판매량에 미치는 영향에 대해 이야기해보자. 
앞으로 알게 되겠지만, 무작위 데이터는 평가를 위해 매우 중요하다. 그러나 현실에서는 무작위 데이터를 수집하는 데에 큰 비용이 든다. (아이스크림의 가격을 무작위로 정해서 팔아보면서 데이터를 모으는 작업은 사실상 불가능할 것이다...) 결과적으로 우리는 처리가 무작위로 할당되지 않은 대다수의 데이터와 아주 약간의 무작위 데이터를 갖게 되는 것이다. 무작위가 아닌 데이터로 모델을 평가하는 것은 매우 까다롭기 때문에, 만약 우리가 약간의 무작위 데이터라도 가지고 있다면 이를 평가 목적으로 남겨두는 것이 좋겠다.
<br>

- 데이터의 형태를 까먹었을까봐, 아래에 테이블 예시를 준비했다.
```python
prices = pd.read_csv("./data/ice_cream_sales.csv") # loads non-random data
prices_rnd = pd.read_csv("./data/ice_cream_sales_rnd.csv") # loads random data
print(prices_rnd.shape)
# (5000, 5)
```
<img src="https://github.com/DoyoungKim12/causal-inference/blob/master/img_BnT/bnt_2_2.PNG?raw=true">

<br>

- 비교를 위해, 2개의 모델을 학습시켜보자. 첫번째 모델은 교차항이 포함된 선형회귀 모델로, 탄력성이 각 유닛별로 차이가 날 수 있다. 

<img src="https://github.com/DoyoungKim12/causal-inference/blob/master/img_BnT/bnt_2_3.PNG?raw=true">

- 모델에 데이터를 적합시키면, 우리는 아래와 같이 탄력성을 예측할 수 있다. 
```python
m1 = smf.ols("sales ~ price*cost + price*C(weekday) + price*temp", data=prices).fit()
```
<img src="https://github.com/DoyoungKim12/causal-inference/blob/master/img_BnT/bnt_2_4.PNG?raw=true">

<br>

- 두번째 모델은 완전히 비모수적인(fully nonparametric) 기계학습 예측 모델이다.
```python
X = ["temp", "weekday", "cost", "price"]
y = "sales"

np.random.seed(1)

# 부스팅트리모델을 사용
m2 = GradientBoostingRegressor()

# 가격 등의 정보로 판매량을 예측
m2.fit(prices[X], prices[y]);
```
<img src="https://github.com/DoyoungKim12/causal-inference/blob/master/img_BnT/bnt_2_5.PNG?raw=true">

- 모델이 과하게 오버피팅되지 않았다는 것을 확인하기 위해, 학습 및 평가에 사용할 데이터에 대해 <img src="https://render.githubusercontent.com/render/math?math=R^2">를 각각 확인할 수 있다. (기계학습에 보다 정통한 사람들을 위해 첨언하자면, 평가 단계에서 성능의 저하가 예상되는데 이는 concept drift(시간이 지남에 따라 모델링 대상의 통계적 특성이 바뀌는 현상)로 인한 것이다. 학습한 데이터는 무작위 데이터가 아니지만, 평가를 위한 데이터는 무작위 데이터이다.)

```python
print("Train Score:", m2.score(prices[X], prices[y]))
print("Test Score:", m2.score(prices_rnd[X], prices_rnd[y]))

# Train Score: 0.9251704824568053
# Test Score: 0.7711074163447711
```

<br>

- 모델을 학습시킨 후, 우리는 회귀 모델에서 탄력성 값을 가져올 것이다. 다시 말하지만, 우리는 근사치를 사용한다. 










