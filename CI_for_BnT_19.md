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

<p align="center">
    <img src="https://render.githubusercontent.com/render/math?math=E[Y|T=1] - E[Y|T=0]"\>
</p>
<p align="center">    
    <img src="https://github.com/DoyoungKim12/causal-inference/blob/master/img_BnT/bnt_2_8.PNG?raw=true">
</p>
<br>

- 위 이미지에서 우리는 첫번째 모델이 판매량을 맞추는 데에는 좋은 성능을 보이지만(실제 판매량과 높은 상관관계를 보임), 해당 그룹은 같은 수준의 처리효과를 보인다. 3개 세그먼트 중 2개는 같은 탄력성을 가지고, 마지막 세그먼트가 다른 그룹에 비해 낮은 탄력성을 가진다. 
<br>

- 반면에, 두번째 모델에서 생성된 그룹은 각기 다른 인과효과를 보인다. (판매량이 가장 많을 것으로 예측된 그룹에서 가격이 할인된 경우의 실제 판매량 차이가 더 크다.) 이는 이 모델이 개인화에도 유용하게 쓰일 수 있다는 신호이다.(판매량이 많을 것으로 예측된 유저에게 할인을 해주면 실제 판매량이 크게 오를 것으로 예상해볼 수 있다.) 마지막으로, 랜덤모델에서 생성된 그룹은 정확히 같은 탄력성을 보인다. 이는 유용하지는 않지만 예상된 결과이다. 
<br>

- 위 그래프를 보는 것만으로도, 어떤 모델이 더 좋은지 감을 잡을 수 있을 것이다. 눈으로 보기에 그룹별 탄력성이 순서에 따라 정렬이 더 잘 되어있고, 각 그룹간 차이가 클수록 더 좋은 모델이라고 할 수 있다. 따라서 여기서는 모델 2가 모델 1보다 낫다. 이를 연속적인 경우로 일반화하기 위해, 우리는 탄력성을 하나의 변수를 사용하는 선형회귀 모델로 추정할 수 있다. 
<p align="center">   
    <img src="https://render.githubusercontent.com/render/math?math=y_i = \beta_0 %2B \beta_1t_i %2B e_i">
</p>

- 만약 한 그룹의 샘플로 모델을 실행시킨다면, 우리는 해당 그룹 내부의 탄력성을 추정하게 된다. 단순 선형회귀의 이론에 의해, 우리는 아래의 식이 성립한다는 사실을 알고 있다. <img src="https://render.githubusercontent.com/render/math?math=\overline{t}">는 처리군 샘플의 처리 평균값, <img src="https://render.githubusercontent.com/render/math?math=\overline{y}">는 결과의 평균치를 의미한다. 

<img src="https://github.com/DoyoungKim12/causal-inference/blob/master/img_BnT/bnt_2_9.PNG?raw=true">

```python
@curry
def elast(data, y, t):
        # line coeficient for the one variable linear regression 
        return (np.sum((data[t] - data[t].mean())*(data[y] - data[y].mean())) /
                np.sum((data[t] - data[t].mean())**2))
```
<br>

- 이제 이를 우리의 아이스크림 가격 데이터에 적용해보자. 이를 위해, 우리는 데이터를 같은 크기(band)로 나누어 탄력도를 계산해줄 함수가 필요하다. 아래의 코드가 도움을 줄 것이다.
```python
def elast_by_band(df, pred, y, t, bands=10):
    return (df
            .assign(**{f"{pred}_band":pd.qcut(df[pred], q=bands)}) # makes quantile partitions
            .groupby(f"{pred}_band")
            .apply(elast(y=y, t=t))) # estimate the elasticity on each partition
```

- 마지막으로, 예측 결과에 의해 구분된 각 그룹별 탄력성을 그래프로 나타내보자. 여기서, 우리는 각 모델을 그룹을 구분하여 각 그룹별 탄력성을 추정하기 위해 사용할 것이다. 
```python
fig, axs = plt.subplots(1, 3, sharey=True, figsize=(10, 4))
for m, ax in enumerate(axs):
    elast_by_band(prices_rnd_pred, f"m{m+1}_pred", "sales", "price").plot.bar(ax=ax)
```
<img src="https://github.com/DoyoungKim12/causal-inference/blob/master/img_BnT/bnt_2_10.PNG?raw=true">

<br>

- 먼저, 랜덤모델(m3)을 보자. 그룹별로 큰 차이 없이 유사한 수준의 탄력성이 관찰된다. 이는 개인화에 그닥 도움이 되지 못하는데, 가격 탄력성이 높은 날과 낮은 날을 구분해주지 못하기 때문이다. 다음으로, 예측모델 m1을 보자. 이 모델은 뭔가 조짐이 좋다! 이 모델은 탄력성이 높은 그룹과 그렇지 않은 그룹을 구분하려고 노력한다. 정확히 우리가 원했던 결과이다.
- 마지막으로, 인과모델 m2는 약간 이상해보인다. 탄력성이 아주 낮은 그룹을 특정하고 있는데, 여기서 낮음의 의미는 가격의 변동에 그만큼 민감함을 의미한다. 이렇듯 가격 변동 민감도가 높은 날을 탐지하는 것은 우리에게 매우 유용하다. 만약 그게 언제인지 우리가 알 수 있다면, 우리는 그 때 가격을 인상하는 것에 주의를 기울일 것이다. 이 모델은 덜 민감한 지역도 특정하긴 하지만, 정렬된 정도는 예측모델만큼 좋지 못하다. 
- 그럼 이제 우리는 무엇을 결정해야 할까? 어떤 모델이 더 유용할까? 예측모델 또는 인과모델? 예측모델의 정렬이 더 나아보이지만, 인과모델은 보다 극적인 민감도를 가지는 그룹을 특정해냈다. 그룹별 탄력도를 체크하는 것이 첫번째 진단으로는 좋지만, 어떤 모델이 나은지 정확히 답하는 기준이 될 수는 없다. 우리는 더 정교한 무언가를 향해 나아가야 한다. 

<br><br>

## Cumulative Elasticity Curv
- 가격이 이진(binary) 처리로 변환되었던 예제를 다시 생각해보자. 그룹(band)별로 처리의 탄력성을 확인할 수 있었다. 우리가 그 다음으로 할 수 있는 것은 각 그룹을 그들이 처리에 민감한 정도에 따라 정렬하는 것이다. 일단 우리가 그룹들을 정렬하고 나면, 누적 탄력성 곡선(Cumulative Elasticity Curve)이라 불리는 것을 구성할 수 있다. 우리는 먼저 첫번째 그룹의 탄력성을 계산하고, 그 다음은 첫번째 그룹과 두번째 그룹의 탄력성을, 이를 반복하여 마지막에는 전체 그룹의 탄력성을 계산하게 된다. 아래의 예시로 좀 더 자세히 살펴보자.

<img src="https://github.com/DoyoungKim12/causal-inference/blob/master/img_BnT/bnt_2_11.PNG?raw=true">

- 누적 탄력성의 첫번째 구간은 가장 민감한 그룹의 ATE와 같다는 사실을 위 그래프에서 확인하자. 또한, 모든 모델에서 누적 탄력성은 전체 데이터셋의 ATE로 수렴한다. 수학적으로는 누적 탄력성을 유닛 i~k까지 추정된 탄력성으로 정의할 수 있다. 

<img src="https://github.com/DoyoungKim12/causal-inference/blob/master/img_BnT/bnt_2_12.PNG?raw=true">

- 누적 탄력성을 구성하기 위해, 우리는 위의 함수를 데이터셋에 반복적으로 수행하여 아래의 시퀀스를 생성한다. 이는 모델 평가 측면에서 매우 흥미로운 것인데, 왜냐하면 우리는 이에 대해 선호를 결정하는 어떤 가이드(preferences statements)를 만들 수 있기 때문이다. 

<img src="https://github.com/DoyoungKim12/causal-inference/blob/master/img_BnT/bnt_2_13.PNG?raw=true">

- 먼저, 모든 k와 0보다 큰 a에 대해 모델은 아래의 부등식을 만족한다. 이는 만약 모델이 탄력성에 따라 각 유닛들을 잘 정렬시켰다면 상위 k개의 샘플에서 관찰되는 탄력성은 상위 k+a개의 샘플에서 관찰되는 탄력성보다 높아야 한다는 것이다. 또는, 단순히 말하자면 내가 상위 유닛을 관찰한다면 그들은 반드시 하위 유닛보다 높은 탄력성을 가져야 한다는 것이다. 

<p align="center">
    <img src="https://render.githubusercontent.com/render/math?math=\hat{y^'}(t)_k > \hat{y^'}(t)_{k %2B a} ">
</p>

- 두번째로, 모델은 모든 k와 0보다 큰 a에 대해 아래의 값이 극대화될 때 더 좋다. 직관적으로 풀이하자면 우리가 원하는 것은 단순히 상위 k개의 유닛이 더 높은 탄력성을 가지는 것 뿐만 아니라 그 차이가 가능한 한 커지는 것이다. 

<p align="center">
    <img src="https://render.githubusercontent.com/render/math?math=\hat{y^'}(t)_k - \hat{y^'}(t)_{k %2B a} ">
</p>

- 이를 보다 구체화하기 위해, 이 아이디어를 코드로 구현해보자. 아래 함수에 대해 몇가지 알아야 될 점이 있다. 먼저, prediction 인자로 전달되는 특정 컬럼에 저장된 값에 의해 결정되는 순서(정렬)를 가정하고 있다. 또한, 첫번째 그룹은 min_period 수만큼의 유닛을 가지고 있어 다른 그룹과 다를 수 있다. 작은 샘플사이즈로 인해 곡선의 시작 부분이 지나치게 noisy할 수 있는 것이다. 이를 수정하기 위해 첫번째 그룹부터 충분한 크기의 샘플을 넘겨주는 것을 고려할 수 있다. 마지막으로, steps 인자로 각 서브그룹마다 얼마나 많은 추가 유닛을 포함시킬지 결정할 수 있다.

```python
def cumulative_elast_curve(dataset, prediction, y, t, min_periods=30, steps=100):
    size = dataset.shape[0]
    
    # orders the dataset by the `prediction` column
    ordered_df = dataset.sort_values(prediction, ascending=False).reset_index(drop=True)
    
    # create a sequence of row numbers that will define our Ks
    # The last item is the sequence is all the rows (the size of the dataset)
    n_rows = list(range(min_periods, size, size // steps)) + [size]
    
    # cumulative computes the elasticity. First for the top min_periods units.
    # then for the top (min_periods + step*1), then (min_periods + step*2) and so on
    return np.array([elast(ordered_df.head(rows), y, t) for rows in n_rows])
```

- 이 함수를 통해, 이제 우리는 각 모델에서 생성된 값을 사용하여 누적 탄력성 곡선을 그래프로 그려볼 수 있다. 누적 탄력성 곡선을 해석하는 것은 어렵지만, 여기에 그 방법이 있다. 이게 오히려 이진 처리의 경우보다 쉬울지도 모른다. X축은 얼마나 많은 수의 샘플을 다루었는지를 의미한다. 여기서는 정규화된 값을 사용했으므로, 0.4의 뜻은 전체 샘플의 40%를 사용했다는 뜻이 된다. Y축은 다수의 샘플에 대해 우리가 예측한 탄력성을 의미한다. 따라서, 만약 곡선이 40%에서 -1의 값을 보인다면 이는 탄력성 상위 40% 유닛들의 그룹 탄력성이 -1이라는 뜻이 된다. 이상적으로 우리가 원하는 것은 가능한 한 많은 수의 샘플에 대해 가장 높은 탄력성이 관찰되는 것이다. 따라서 이상적인 곡선의 형태는 Y축의 최상단에서 시작하여 평균 탄력성 값까지 매우 완만하게 감소하는 것으로, 이 경우에는 평균 탄력도 이상의 유닛을 어느 정도 남겨두고도 높은 비율의 유닛에 처리를 가할 수 있게 될 것이다. 

<img src="https://github.com/DoyoungKim12/causal-inference/blob/master/img_BnT/bnt_2_14.PNG?raw=true">

- 더 말할 필요도 없이, 3개 모델 중 어느 것도 이상적인 형태의 탄력성 곡선 근처에도 가지 못했다. 랜덤모델인 M3는 평균 탄력성 값 주변에서 진동하는 형태를 보여준다. 이는 모델이 평균 탄력도와는 다른 탄력도 값을 가지는 그룹을 찾지 못했음을 의미한다. 예측모델인 M1은 역순으로 정렬된 탄력도를 보여주는 듯 하다. 곡선이 평균 탄력성 값의 아래에서 시작하기 때문이다. 그 뿐만이 아니라, 평균치까지 빠르게 수렴하는 모습을 보여준다. 마지막으로, 인과모델인 M2는 더 흥미롭다. 누적 탄력도가 평균치로부터 점점 증가하는 모습을 보이다가 75%의 유닛을 다루는 특정 포인트에 도달하고나서 거의 0에 가까운 탄력도를 유지한다. 이는 아마도 이 모델이 아주 낮은 탄력도, 즉 가격에 매우 민감한 시기를 구분할 수 있기 때문일 것이다. 따라서 그 특정 시기에 가격을 인상하지 않는다면, 우리는 75%를 차지하는 다수의 시기에 대하여 가격 인상을 할 수도 있을 것이다. (다른 시기는 낮은 가격 민감도를 보일 것이므로)

- 모델 평가의 관점에서, 누적 탄력성 곡선은 이미 그룹별 탄력도를 관찰하는 단순한 아이디어보다 훨씬 나아보인다. 여기서 우리는 보다 정밀한 모델 평가 방법을 만들어내기 위해 노력했다. 여전히, 이 곡선을 직관적으로 이해하기는 어렵다. 그렇기 때문에 우리는 더 나아진 개선안 하나를 더 해볼 수 있다.

<br><br>

## Cumulative Gain Curve
- 다음 아이디어는 매우 간단하지만 강력한 개선안이다. 우리는 누적 탄력성 값에 샘플의 크기를 곱할 것이다. 예를 들어, 누적 탄력성 값이 40% 지점에서 -0.5라고 한다면 40% 지점의 값을 0.4에 -0.5를 곱한 -0.2로 하겠다는 것이다. 그러고나서, 우리는 이를 랜덤모델의 곡선과 비교해볼 것이다. 이 곡선은 실제로 0에서부터 시작하여 평균 처리 효과(ATE) 수준에 점점 가까워지는 직선이다. 이렇게 생각해보자. 랜덤모델의 누적 탄력성에서 모든 값은 ATE인데, 왜냐하면 이 모델은 데이터를 랜덤하게 나누기 때문이다. 원점을 지나고 기울기가 1인 직선에 ATE를 곱하면 기울기가 ATE가 되어 (0,0)과 (1,ATE)를 지나는 직선을 관찰할 수 있다. 

<img src="https://github.com/DoyoungKim12/causal-inference/blob/master/img_BnT/bnt_2_15.PNG?raw=true">

- 일단 이론적인 랜덤모델의 곡선을 정의하고 나면, 우리는 이를 벤치마크로 활용하여 우리의 모델을 이와 비교해볼 수 있다. 모든 곡선은 같은 곳에서 시작해 같은 곳에서 끝난다 (각각 0, ATE) 그러나, 모델이 탄력성 기준으로 각 유닛들을 더 잘 정렬할수록 0과 1 사이의 어떤 값으로 발산해나가는 모습을 관찰할 수 있다. 예를 들어, M2는 M1보다 좋은 모델이라고 할 수 있는데 이는 끝부분의 ATE 값에 도달하기 전까지 더 큰 값으로 발산하기 때문이다. ROC 커브에 익숙한 사용자는 누적 이득(Cumulative Gain)을 인과모델의 ROC로 생각할 수 있다. 

- 수학적으로 설명하자면 아래와 같다.

<img src="https://github.com/DoyoungKim12/causal-inference/blob/master/img_BnT/bnt_2_16.PNG?raw=true">

- 이를 코드에 적용하기 위해 우리가 해야할 것은 단순히 비례 샘플 사이즈로 정규화하는 부분 뿐이다. 
```python
def cumulative_gain(dataset, prediction, y, t, min_periods=30, steps=100):
    size = dataset.shape[0]
    ordered_df = dataset.sort_values(prediction, ascending=False).reset_index(drop=True)
    n_rows = list(range(min_periods, size, size // steps)) + [size]
    
    ## add (rows/size) as a normalizer. 
    return np.array([elast(ordered_df.head(rows), y, t) * (rows/size) for rows in n_rows])
```

- 우리의 아이스크림 데이터에 적용하면 아래와 같은 곡선들을 관찰하게 된다. 이제 인과모델(M2)가 다른 두 모델에 비해 훨씬 낫다는 것이 명확하게 보인다. M2의 곡선은 랜덤 라인에서 M1과 M3에 비해 훨씬 많이 발산한다. 이것으로, 우리는 인과모델을 평가하는 정말 멋진 아이디어를 다뤄보았다. 이것만으로도 큰 발전이다. 우리는 탄력성 순으로 정렬하는 모델이 얼마나 정확한지를, 단순한 정답(ground truth)없이도 해내려고 노력했다. 이제 남은 것은 단 한가지로, 이러한 측정치에 신뢰구간을 포함하는 것이다. 어쨌든 우리가 야만인은 아니지 않은가? (신뢰구간 없는 추정치에 대한 경계를 나타내는 말인듯...)

<img src="https://github.com/DoyoungKim12/causal-inference/blob/master/img_BnT/bnt_2_17.PNG?raw=true">

<br><br>

## Taking Variance Into Account
- 탄력성 곡선을 다룰 때 편차를 고려하지 않는 것은 잘못된 것 같다. 특히 모두 선형 회귀 이론을 사용하기 때문에, 신뢰 구간을 추가하는 것은 상당히 쉬울 것이다. 신뢰구간을 구하기 위해, 우리는 먼저 선형회귀 파라미터(계수)에 대한 CI를 리턴하는 함수를 하나 만들 것이다. 나는 단순 선형회귀의 공식을 사용했지만, CI를 도출하는 데에는 원하는 다른 공식을 사용해도 무방하다. cumulative_elast_curve 함수에 약간의 수정을 가하여 탄력성의 신뢰구간을 계산할 수 있고, 아래의 그래프로 M2 모델로 구한 탄력성 곡선의 95% 신뢰구간을 확인할 수 있다. 

<img src="https://github.com/DoyoungKim12/causal-inference/blob/master/img_BnT/bnt_2_18.PNG?raw=true">

```python
def elast_ci(df, y, t, z=1.96):
    n = df.shape[0]
    t_bar = df[t].mean()
    beta1 = elast(df, y, t)
    beta0 = df[y].mean() - beta1 * t_bar
    e = df[y] - (beta0 + beta1*df[t])
    se = np.sqrt(((1/(n-2))*np.sum(e**2))/np.sum((df[t]-t_bar)**2))
    return np.array([beta1 - z*se, beta1 + z*se])
```

```python
def cumulative_elast_curve_ci(dataset, prediction, y, t, min_periods=30, steps=100):
    size = dataset.shape[0]
    ordered_df = dataset.sort_values(prediction, ascending=False).reset_index(drop=True)
    n_rows = list(range(min_periods, size, size // steps)) + [size]
    
    # just replacing a call to `elast` by a call to `elast_ci`
    return np.array([elast_ci(ordered_df.head(rows), y, t)  for rows in n_rows])
```

<img src="https://github.com/DoyoungKim12/causal-inference/blob/master/img_BnT/bnt_2_19.PNG?raw=true">

- 데이터셋의 크기가 증가할수록 CI가 점점 작아지는 것을 확인할 수 있다. 
- 누적 이득 곡선에 대해서도 같은 방식으로 간단히 CI를 구할 수 있다. elast 함수를 elast_ci 함수로 변경하기만 하면 된다. 아래 그래프도 역시 M2 모델의 곡선이다. 주목할 점은 CI 값이 샘플사이즈가 작은 초반 구간에서 더 작다는 것이다. 그 이유는 정규화 과정에서 N으로 나눌때 그 과정이 ATE 파라미터를 축소시키고 CI도 이와 관련이 있기 때문이다. 그리고 이는 당연히 모든 모델에 동일하게 적용되기 때문에 모델간 비교에는 전혀 문제가 되지 않는다.  

```python
def cumulative_gain_ci(dataset, prediction, y, t, min_periods=30, steps=100):
    size = dataset.shape[0]
    ordered_df = dataset.sort_values(prediction, ascending=False).reset_index(drop=True)
    n_rows = list(range(min_periods, size, size // steps)) + [size]
    return np.array([elast_ci(ordered_df.head(rows), y, t) * (rows/size) for rows in n_rows])
```

<br><br>

## Key Ideas
- 우리는 이 챕터에서 탄력성 순으로 정렬하는 것에 있어 모델의 성능을 평가하는 3가지 방법에 대해 알아보았다. 인과추론 목적의 모델간 비교를 위해 사용한 것이고, 이는 쉽지 않았다. 우리는 모델이 다른 탄력성 값을 가지는 그룹을 잘 찾아낸다면, 실제로 탄력성을 관찰할 수 없더라도 모델의 평가가 가능한지 알아보려고 한 것이다. 

- 우리는 여기서 무작위 데이터에 굉장히 많이 의존했다. 우리는 모델을 무작위로 처리가 할당되지 않은 데이터로 훈련시켰지만, 모든 평가는 처리가 무작위로 할당된 데이터로 이루어졌다. 이는 탄력성을 보다 신뢰성있게 평가하는 방법이 필요했기 때문이다. 랜덤데이터가 없다면 우리가 사용한 이 간단한 공식들은 쓸모가 없게 될 것이다. 지금은 매우 잘 알다시피, 단순한 선형회귀는 교란 변수가 존재할 때 변수 편향(variable bias)를 생략해왔다. 

- 그럼에도 불구하고, 우리가 약간의 무작위 데이터라도 얻을 수 있게 된다면, 우리는 이미 랜덤모델을 비교하는 방법을 알고 있다. 다음 챕터에서, 우리는 무작위가 아닌 데이터를 다룰 것이지만, 시작하기 전에 나는 모델 평가에 대해 몇 마디 첨언하고자 한다.

- 믿을 수 있는 모델 평가가 얼마나 중요한지 다시 곱씹어보자. 누적 이득 곡선으로, 우리는 드디어 인과추론을 위해 만들어진 모델을 평가하는 좋은 방법을 찾았다. 우리는 이제 어떤 모델이 처리 개인화에 더 나은 성능을 보이는지 결정할 수 있다. 중요한 건 이것이다. 인과추론에 대해 찾을 수 있는 대부분의 자료에서는 모델을 평가하는 좋은 방법을 제시하지 않는다. 내 생각에는, 이것이야말로 인과추론을 머신러닝만큼 유명하게 만들기 위해 필요한 부분(missing piece)이다. 좋은 평가방법이 있다면, 우리는 인과추론을 이미 예측모델에서 아주 유용하게 쓰이고 있는 train-test 패러다임에 좀더 가까이 적용할 수 있다. 이는 대담한 발언이다. 그 말은 내가 조심스럽게 말하고 있다는 뜻이지만, 지금까지 이에 대한 합당한 비판을 발견하지 못한 것도 사실이다. 혹시 있으면 알려달라.





