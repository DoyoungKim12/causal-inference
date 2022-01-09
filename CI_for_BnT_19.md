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

<img src="https://github.com/DoyoungKim12/causal-inference/blob/master/img_BnT/bnt_2_6.PNG?raw=true">

- 두 모델은 무작위가 아닌 데이터로 학습되었다. 이제 우리는 무작위 데이터로 돌아가 예측결과를 생성할 것이다. 우리가 모든 결과물을 한 곳에 모을 수 있도록, 우리는 하나의 데이터프레임에 머신러닝 모델의 예측결과와 인과 모델의 탄력성 예측결과를 추가할 것이다. 그리고 랜덤 모델 하나를 더 추가하자. 이 모델은 예측값으로 무작위 값을 리턴하여, 벤치마크로 기능할 것이다. 
```python
def predict_elast(model, price_df, h=0.01):
    return (model.predict(price_df.assign(price=price_df["price"]+h))
            - model.predict(price_df)) / h

np.random.seed(123)
prices_rnd_pred = prices_rnd.assign(**{
    "m1_pred": m2.predict(prices_rnd[X]), ## predictive model
    "m2_pred": predict_elast(m1, prices_rnd), ## elasticity model
    "m3_pred": np.random.uniform(size=prices_rnd.shape[0]), ## random model
})

prices_rnd_pred.head()
```
<img src="https://github.com/DoyoungKim12/causal-inference/blob/master/img_BnT/bnt_2_7.PNG?raw=true">

<br><br>

## Elasticity by Model Band
- 이제 우리는 우리가 예측한 결과를 가지고 있고, 그 값들이 얼마나 정확한지를 평가해야 한다. 그리고 우리는 탄력성을 관찰할 수는 없으므로 우리가 비교할 단순한 정답(ground truth)은 
존재하지 않는다. 대신, 우리의 탄력성 모델로부터 우리가 원했던 것이 무엇인지 다시 생각해보자. 아마도 그것이 우리가 그 값을 어떻게 평가해야 하는지에 대한 단서가 될 것이다. 
<br>

- 처리 탄력성 모델을 만드는 아이디어는 어떤 유닛이 처리에 더 민감하게, 혹은 둔감하게 반응하는지를 찾고자 하는 필요에서 시작되었다. 즉, 개인화를 원했기 때문에 시작되었던 것이다. 어떤 마케팅 캠페인은 오직 하나의 세그먼트에서만 매우 유효할지도 모른다. 할인도 오직 특정 타입의 고객에게만 작동할지도 모른다. 좋은 인과 모델은 어떤 고객이 주어진 처리에 대해 더 많이, 또는 적게 반응하는지 찾는 일을 도와야 한다. 우리의 아이스크림 예제에서, 모델은 어떤 날에 사람들이 아이스크림에 더 많은 돈을 기꺼이 지불하는지, 또는 어떤 날에 가격 탄력성이 덜 부정적으로 작용하는지를 알아내야만 하는 것이다. 
<br>

- 만약 이것이 목표라면, 어떻게든 각 유닛들을 가장 예민한 유닛부터 덜 예민한 유닛 순으로 정렬할(order) 수만 있다면 매우 유용할 것이다. 우리는 예측된 탄력성 수치를 가지고 있기 때문에, 우리는 각 유닛을 예측에 근거하여 정렬할 수 있고, 그것이 실제 탄력성에 근거한 정렬이길 기대한다. 슬프게도 그 정렬된 순서의 정확도를 각 유닛 레벨에서 평가할 수는 없다. 그러나, 그럴 필요가 없다면 어떨까? 만약, 그 대신 우리가 그 정렬된 순서에 의해 정의된 그룹을 평가하게 된다면? 만약 우리의 처리가 랜덤하게 분포되었다면(여기가 바로 무작위성이 등장하는 부분이다!), 유닛들로 이루어진 그룹의 탄력성을 평가하는 것은 쉽다. 우리에게 필요한 것은 처리군과 대조군의 결과를 비교하는 것 뿐이다. 
<br>

- 이를 더 잘 이해하기 위해, 이항 처리가 이루어지는 경우를 시각화해보자. 가격설정 예시를 그대로 가져가되, 이제는 처리가 (가격 조정이 아닌) 단순 할인이 된다. 다시 말해, 가격은 높거나(대조군) 낮거나(처리군) 둘 중 하나이다. 실제 판매량을 Y축에 두고 그래프를 그려보자. X축은 각 모델에 의해 예측된 판매량이고, 각 점의 색은 가격을 나타낸다. 그러고나서, 데이터를 3개의 같은 크기의 그룹으로 나누자. **만약 처리가 무작위로 할당되었다면,** 우리는 쉽게 각 그룹의 ATE를 계산할 수 있다. 

<img src="https://render.githubusercontent.com/render/math?math=E[Y|T=1] - E[Y|T=0]">
<img src="https://github.com/DoyoungKim12/causal-inference/blob/master/img_BnT/bnt_2_8.PNG?raw=true">

<br>

- 위 이미지에서 우리는 첫번째 모델이 판매량을 맞추는 데에는 좋은 성능을 보이지만(실제 판매량과 높은 상관관계를 보임), 해당 그룹은 같은 수준의 처리효과를 보인다. 3개 세그먼트 중 2개는 같은 탄력성을 가지고, 마지막 세그먼트가 다른 그룹에 비해 낮은 탄력성을 가진다. 
<br>

- 반면에, 두번째 모델에서 생성된 그룹은 각기 다른 인과효과를 보인다. (판매량이 가장 많을 것으로 예측된 그룹에서 가격이 할인된 경우의 실제 판매량 차이가 더 크다.) 이는 이 모델이 개인화에도 유용하게 쓰일 수 있다는 신호이다.(판매량이 많을 것으로 예측된 유저에게 할인을 해주면 실제 판매량이 크게 오를 것으로 예상해볼 수 있다.) 마지막으로, 랜덤모델에서 생성된 그룹은 정확히 같은 탄력성을 보인다. 이는 유용하지는 않지만 예상된 결과이다. 
<br>

- 위 그래프를 보는 것만으로도, 어떤 모델이 더 좋은지 감을 잡을 수 있을 것이다. 눈으로 보기에 그룹별 탄력성이 순서에 따라 정렬이 더 잘 되어있고, 각 그룹간 차이가 클수록 더 좋은 모델이라고 할 수 있다. 따라서 여기서는 모델 2가 모델 1보다 낫다. 이를 연속적인 경우로 일반화하기 위해, 우리는 탄력성을 하나의 변수를 사용하는 선형회귀 모델로 추정할 수 있다. 

<img src="https://render.githubusercontent.com/render/math?math=y_i = \beta_0  \beta_1t_i  e_i">

- 만약 한 그룹의 샘플로 모델을 실행시킨다면, 우리는 해당 그룹 내부의 탄력성을 추정하게 된다. 단순 선형회귀의 이론에 의해, 우리는 아래의 식이 성립한다는 사실을 알고 있다. <img src="https://render.githubusercontent.com/render/math?math=\overline{t}">는 처리군 샘플의 처리 평균값, <img src="https://render.githubusercontent.com/render/math?math=\overline{y}">는 결과의 평균치를 의미한다. 

<img src="https://github.com/DoyoungKim12/causal-inference/blob/master/img_BnT/bnt_2_9.PNG?raw=true">

```python
@curry
def elast(data, y, t):
        # line coeficient for the one variable linear regression 
        return (np.sum((data[t] - data[t].mean())*(data[y] - data[y].mean())) /
                np.sum((data[t] - data[t].mean())**2))
```

















