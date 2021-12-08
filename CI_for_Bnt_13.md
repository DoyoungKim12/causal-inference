<br><br><br><br>

# 13 - Panel Data and Fixed Effects

<br><br>

## Controlling What you Cannot See
- 성향점수(propensity score), 선형회귀, 매칭 등의 방법론은 무작위가 아닌 데이터에서의 교란(confounding)을 제어하는 좋은 수단이지만, 모두 **조건부 비교란성(conditional unconfoundedness)** 이라는 하나의 중요한 가정에 의존한다. ($(Y_0,Y_1 \perp T \bar X)$)
- 말로 풀어서 설명하자면, 모든 교란변수들을 우리가 알고 있고 측정가능해서 그들을 모두 조건화하여 처리 수준이 무작위로 배정된 것과 같이 되어야 한다는 것이다. 이에 대한 중요한 이슈 중 하나는, 가끔 우리가 교란변수를 알고도 측정하지 못한다는 데에 있다. 
  - 예시 : 교육수준, 결혼, 외모가 수입에 미치는 영향
    - 남자의 수입에 결혼이 미치는 영향을 밝혀내는 문제에 대해 생각해보자. 기혼 남성이 더 많은 돈을 번다는 것은 경제학에서 잘 알려진 사실이다. 그러나, 이러한 관계성이 인과성으로 이어지는지는 확실하지 않다. 교육을 더 많이 받은 사람이 결혼도 많이 하고 수입도 높을 가능성이 있는 것이다. (즉, 교육수준이 결혼이 수입에 미치는 영향에 대한 교란변수가 된다.) 이러한 교란변수에 대해, 우리는 교육수준을 츶겅하고 이를 조건화한 회귀식을 세울 수 있었다. 
    - 하지만 또다른 교란변수로 외모가 있을 수 있다. 얼굴이 잘생길수록 결혼할 가능성과 더 ㅁ많은 돈을 벌 가능성이 높을 것이다. 그러나, 외모는 지능처럼 우리가 쉽게 측정할 수 없는 무언가이다.

<br><br>

- 이는 우리를 어려운 상황에 처하게 한다. 우리가 측정되지 못한 교란변수를 가지고 있다면, 이는 곧 편향이 있음을 의미하기 때문이다. 이를 해결하는 방법은 이전에 보았던 것처럼 도구변수를 사용하는 방법이 있긴 하다. 그러나 좋은 도구변수를 떠올리는 것은 쉽지 않을 뿐더러 꽤 높은 수준의 창의력을 필요로 한다. 따라서, 다른 대안을 살펴보자. 
- 아이디어는 **패널 데이터**를 사용하자는 것이다. 패널 데이터는 같은 개인에 대하여 각 시기에 따른 다수의 데이터를 관찰한 것을 말한다. 패널 데이터 형식은 현업에서는 매우 흔한데, 그들이 고객의 행동 데이터를 시기마다 보관하기에 그렇다. 패널 데이터를 우리가 활용할 수 있는 이유는 개인별로 처리 전후를 관찰하여 그들이 어떻게 행동했는지를 확인할 수 있기 때문이다. 수학의 세계로 들어가기 이전에, 이게 어떤 직관을 주는지를 먼저 살펴보자.

<br><br>

- 먼저, 아래의 인과그래프를 보자. 같은 유닛에 대한 시계열 데이터이다. 첫 번째(1기) 결혼이 동시에 소득과 그 이후의 결혼 상태를 야기하는 상황을 가정해보자. 이러한 관계성은 2기와 3기에도 마찬가지다. 또한, 외모는 모든 시기 전반에 관여하며 결혼과 수입에 영향을 준다고 가정한다.
<img src="https://github.com/DoyoungKim12/causal-inference/blob/master/img_BnT/bnt_32.PNG?raw=true">

<br><br>

- 우리는 외모라는 변수를 측정할 수 없기 때문에 조건화할 수 없다. 그러나 우리는 패널 구조를 사용하기 때문에 이제 이는 더이상 문제가 되지 않는다. 아이디어는 외모와 같은 다른 모든 속성들이 시간의 흐름에도 일정하다고 보는 것이다. 따라서 우리가 그들을 직접 조건화할 수 없더라도, 우리는 이를 개인 자체의 데이터를 보는 것으로 통제할 수 있다. 
<img src="https://github.com/DoyoungKim12/causal-inference/blob/master/img_BnT/bnt_33.PNG?raw=true">

<br><br>

- 생각해보자. 우리는 외모나 지능과 같은 속성을 측정할 수 없지만 그러한 사람의 속성은 시간이 지나도 동일하다는 것을 안다. 따라서, 우리는 개인을 특정하는 더미 변수를 생성하여 이를 선형 모델에 추가할 수 있다. 이것이 바로 우리가 개인 그 자체로 조건화한다는 것의 의미이다. 개인 더미 변수를 모델에 추가한 상태에서 수입에 대한 결혼의 영향을 측정할 때, 회귀 모델은 개인 변수를 고정한 상태에서 결혼의 효과를 추정한다. 이처럼 개인 변수(entity dummy)를 추가한 것을 수정된 효과 모델(fixed effect model)이라고 부른다.

<br><br>

## Fixed Effects
- 문제를 보다 정형화하기 위해, 우리가 가진 데이터를 보자. 우리가 가진 예시를 따라, 결혼이 수입에 미치는 효과를 추정해볼 것이다. 데이터는 이 2개 변수(결혼과 임금)를 포함하는데, 다수의 개인에 대한 다년간의 데이터 또한 포함한다. (여기서 임금은 로그변환된 값이다.) 추가로, 우리는 조건화할 대상들 또한 가지고있다. (일한 시간, 학력 등...)

```python
from linearmodels.datasets import wage_panel
data = wage_panel.load()
data.head()
```
<img src="https://github.com/DoyoungKim12/causal-inference/blob/master/img_BnT/bnt_34.PNG?raw=true">

<br><br>

- 일반적으로, 수정된 효과 모델은 아래와 같이 정의된다.
 $$y_{it} = \betaX_{it} + \gammaU_i + e_{it}$$
 
- $y_{it}$는 개인 i의 t기 결과이다.
- $X_{it}$는 t기의 개인 i에 대한 변수 벡터이다. (관찰가능한 결혼 또는 경험 등)
- $U_{i}$는 개인 i에 대해 관찰되지 않은 것들(unobservables)이다. (외모, 지능 등)
  - 관찰되지 않는 것들은 시간에 따라 변하지 않기 때문에 t가 생략된다.
- $e_{it}$는 오차항이다.

<br><br>

- 이제, 수정된 효과 모델에 패널 데이터를 사용하는 것은 개인에 대한 더미 변수를 더하는 것처럼 단순한 것이라고 했던 나의 말을 다시 떠올려보자. 맞는 말이긴 하지만, 실제로는 우린 그렇게 하진 않을 것이다. 100만명의 고객이 있는 데이터셋을 상상해보라. 각각에 대한 더미를 더한다면, 1백만개의 열이 생겨나게 될 것이고 이는 아마 좋은 생각이 아닐 것이다. 대신, 우리는 선형회귀를 2개의 분리된 모델로 나누는 트릭을 적용할 것이다. 이 트릭은 이전에 보았던 것이지만 한번 더 요약하기 좋은 타이밍이 되었다.
- 어떤 feature들의 집합인 $X_1$과 다른 feature들의 집합인 $X_2$로 이루어진 선형회귀 모델이 하나 있다고 가정하자.

$$\hat{Y} = \hat{\beta_1}X_1 + \hat{\beta_2}X_2$$

- $X_1$과 $X_2$는 feature의 행렬로, 행은 feature, 열은 관측된 값들(observation)이다.
- $\beta_1$과 $\beta_2$는 행백터이다. 아래와 같은 방법으로 정확히 같은 $\beta_1$을 얻을 수 있다.

1. 결과 y에 대한 회귀식을 두번째 집합의 feature들로만 구성한다. ($\hat{y^\*} = \hat{\gamma_1}X_2$)
2. 첫번째 집합의 feature에 대한 회귀식을 두번째 집합으로 구성한다. ($\hat{X_1} = \hat{\gamma_2}X_2$)
3. 1과 2의 잔차를 구한다. ($\tilde{X_1} = X_1 - \hat{X_1}$, $\tilde{y_1} = y_1 - \hat{y^\*}$)
4. y 잔차에 대한 회귀식을 첫번째 집합 feature의 잔차로 구성한다. ($\tilde{y} = \hat{\beta_1}\tilde{X_1}$)

<br><br>

- 마지막 회귀에서의 계수는 모든 feature를 사용한 회귀식에서와 같을 것이다. 그런데 이게 어떻게 우리에게 도움이 된다는 것일까? 우리는 개인 더미변수를 사용한 모델의 추정치를 2개로 나눌 수 있다. 먼저, 우리는 더미변수들을 각각 결과와 다른 feature들을 예측하는 데에 사용했다. 이것이 각각 위의 1단계와 2단계이다.
- 이제, 더미 변수로 회귀식을 구성하는 것이 그 더미에 대한 평균을 추정하는 것만큼 단순하다는 것을 확인해보자. 우리의 데이터가 그게 진짜라는 사실을 보여줄 것이다. 연도를 더미변수로 하여, 그 변수로 구성한 함수로 임금을 예측하는 모델을 만들어보자.

```python
mod = smf.ols("lwage ~ C(year)", data=data).fit()
mod.summary().tables[1]
```
<img src="https://github.com/DoyoungKim12/causal-inference/blob/master/img_BnT/bnt_35.PNG?raw=true">

- 이 모델이 평균 임금을 어떻게 예측하고 있는지 보자. 1980년에는 1.3935, 81년에는 1.3935+0.1194 = 1.5129 와 같이 예측되고 있다. 이제 우리가 연도별 평균을 계산한다면 정확히 같은 결과를 얻게 된다. 

```python
data.groupby("year")["lwage"].mean()
```
<img src="https://github.com/DoyoungKim12/causal-inference/blob/master/img_BnT/bnt_36.PNG?raw=true">

- 이는 우리가 패널 데이터의 모든 개인에 대해 평균을 구한다면, 우리는 본질적으로 개인 더미 변수에 대한 회귀식을 다른 변수들로 구성하는 것이 된다는 것을 의미한다. 이는 아래의 추정 과정을 이끌어내는 모티브가 된다.

<img src="https://github.com/DoyoungKim12/causal-inference/blob/master/img_BnT/bnt_37.PNG?raw=true">

- 첫번째로 각 개인별 평균을 빼서 시간에 종속적인 변수를 만든다. 이는 독립변수와 종속변수 모두에 적용되고, 두번째로는 시간 종속적인 변수들로 회귀식을 구성하는데 여기서 가장 중요한 내용이 등장한다. 이렇게 하면 관찰되지 않은 $U_i$를 제거할 수 있다는 것이다. 왜냐하면 $U_i$는 시간의 흐름에 따라 변하지 않는 값이기 때문에, 시간의 흐름에 따른 변동치에 대한 회귀식에서는 시간의 흐름에 따라 변하지 않는 변수의 영향력이 제거되기 때문이다. 
- 그리고 당연히 관찰되지 않은 변수만 제거되는 것은 아니다. 시간의 흐름에 따라 변하지 않는 모든 변수가 제거된다. 따라서, 그런 변수들은 애초에 모델에 적용하는 것이 불가능하다.

<img src="https://github.com/DoyoungKim12/causal-inference/blob/master/img_BnT/bnt_38.PNG?raw=true">

<br><br>

- 어떤 변수가 위와 같은 특성을 가지는지 체크하기 위해, 데이터를 개인별로 그룹화하고 각 그룹의 표준편차 합계를 구할 수 있다. 만약 이 값이 0이라면, 시간에 따라 변하지 않는 값으로 해석할 수 있다. 
```python
data.groupby("nr").std().sum()
```
<img src="https://github.com/DoyoungKim12/causal-inference/blob/master/img_BnT/bnt_39.PNG?raw=true">

- 우리의 데이터에서는 인종과 관련된 더미변수(black, hisp)를 제거해야 하는데, 이는 개인에 대해 시간의 변화와 관계없이 일정한 값을 가지기 때문이다, 또한, 우리는 education 변수도 제거해야한다. 직업 변수도 사용하지 않는 것이 좋은데, 해당 변수는 아마도 결혼이 임금에 미치는 영향을 중재하는 변수(mediator)일 것이다. 우리가 사용할 변수를 골랐으니, 이제는 모델로 효과를 추정할 시간이다. 
- 모델을 구성하기 위해, 먼저 개인별 평균값 데이터를 가져오자. 개인별로 모든 변수를 그룹화하고 그 안에서 평균값을 구하여 얻을 수 있다.

```python
Y = "lwage"
T = "married"
X = [T, "expersq", "union", "hours"]

mean_data = data.groupby("nr")[X+[Y]].mean()
mean_data.head()
```
<img src="https://github.com/DoyoungKim12/causal-inference/blob/master/img_BnT/bnt_40.PNG?raw=true">

- 평균만큼의 값을 빼주기(demean) 위해, 원본 데이터의 인덱스가 개인 식별자가 되도록 설정해야 한다. 그러면 단순히 데이터프레임을 빼줌으로써 평균만큼의 값을 뺄 수 있다.

```python
demeaned_data = (data
                 .set_index("nr") # set the index as the person indicator
                 [X+[Y]]
                 - mean_data) # subtract the mean data

demeaned_data.head()
```
<img src="https://github.com/DoyoungKim12/causal-inference/blob/master/img_BnT/bnt_41.PNG?raw=true">

- 마지막으로, 시간 종속적인 이 데이터(time-demeaned data)로 수정된 효과 모델(fixed effect model)을 구성할 수 있다.

```python
mod = smf.ols(f"{Y} ~ {'+'.join(X)}", data=demeaned_data).fit()
mod.summary().tables[1]
```
<img src="https://github.com/DoyoungKim12/causal-inference/blob/master/img_BnT/bnt_42.PNG?raw=true">

- 만약 우리가 수정된 효과 모델이 모든 누락된 변수들로 인한 편향을 제거한다고 믿는다면, 이 모델은 결혼이 남성의 임금을 11% 가량 높인다고 말해주고 있다. 이 결과는 매우 중요하다. 여기서 자세히 봐야할 점 하나는 표준오차가 클러스터화되어야 한다는 것이다. 따라서, 우리의 모든 추정을 손으로 일일히 하는 대신, 우리는 linearmodels 라이브러리를 사용하여 cluster_entity라는 argument를 True 값으로 설정할 수 있다.

```python
from linearmodels.panel import PanelOLS
mod = PanelOLS.from_formula("lwage ~ expersq+union+married+hours+EntityEffects",
                            data=data.set_index(["nr", "year"]))

result = mod.fit(cov_type='clustered', cluster_entity=True)
result.summary.tables[1]
```
<img src="https://github.com/DoyoungKim12/causal-inference/blob/master/img_BnT/bnt_43.PNG?raw=true">

- 여기서의 계수 추정치가 우리가 이전에 얻었던 값과 일치하는 것을 확인하자. 달라진 점은 표준 오차가 조금 더 커졌다는 점이다. 이제 이 결과를 시계열을 고려하지 않은 단순 OLS 모델과 비교해보자. 이를 위해, 시간의 변화에 일정한 변수들을 다시 넣을 것이다.

```python
mod = smf.ols("lwage ~ expersq+union+married+hours+black+hisp+educ", data=data).fit()
mod.summary().tables[1]
```
<img src="https://github.com/DoyoungKim12/causal-inference/blob/master/img_BnT/bnt_44.PNG?raw=true">

- 이 모델은 결혼이 남성의 임금을 14% 증가시켰다고 말한다. 우리가 수정된 효과모델로 추정한 값보다 큰 수치이다. 이는 지능, 외모와 같은 고정된 개별 요인으로 인해 생략된 일부 변수의 편향이 모델에 더해지지 않았음을 나타낸다. 

<br><br>

## Visualizing Fixed Effects
- 수정된 효과모델이 어떻게 동작하는지에 대한 우리의 직관을 확장하기 위해, 다른 예제를 좀 더 들여다보자. 당신이 빅테크 기업에서 일하고, 인앱 구매에 옥외광고 마케팅 캠페인이 미친 효과를 측정하려고 한다고 가정해보자. 과거의 데이터를 볼 때, 당신은 마케팅 부서가 구매 레벨이 낮은 도시의 옥외광고에 더 많은 돈을 사용하는 경향이 있다는 사실을 알아냈다. 이게 옳은 결정일까? 매출이 급증했다면 그렇게나 많은 광고를 할 필요도 없었을 것이다. 이 데이터에 대해 회귀식을 세운다면, 마케팅에 높은 비용을 지불할수록 인앱 구매가 낮아지는 것처럼 나올 것이다. 그리고 그 이유는 마케팅 투자가 소비가 적은 지역에 편향되어있기 때문이다.

```python
toy_panel = pd.DataFrame({
    "mkt_costs":[5,4,3.5,3, 10,9.5,9,8, 4,3,2,1, 8,7,6,4],
    "purchase":[12,9,7.5,7, 9,7,6.5,5, 15,14.5,14,13, 11,9.5,8,5],
    "city":["C0","C0","C0","C0", "C2","C2","C2","C2", "C1","C1","C1","C1", "C3","C3","C3","C3"]
})

m = smf.ols("purchase ~ mkt_costs", data=toy_panel).fit()

plt.scatter(toy_panel.mkt_costs, toy_panel.purchase)
plt.plot(toy_panel.mkt_costs, m.fittedvalues, c="C5", label="Regression Line")
plt.xlabel("Marketing Costs (in 1000)")
plt.ylabel("In-app Purchase (in 1000)")
plt.title("Simple OLS Model")
plt.legend();
```
<img src="https://github.com/DoyoungKim12/causal-inference/blob/master/img_BnT/bnt_44.PNG?raw=true">

<br><br>

- 인과추론에 대해 잘 알고 있었던 당신은 수정된 효과 모델을 사용하기로 결정한다. 모델에 각 도시의 더미변수를 추가하기로 한 것이다. 이 모델은 시간의 흐름에 일정한 각 도시의 특징을 제어하기 때문에, 만약 어떤 한 도시가 구매율이 좀 낮더라도 우리는 이를 포착할 수 있다. 모델을 실행해본다면, 마케팅 비용이 높을수록 인앱 구매도 늘어나는 것을 확인할 수 있다.
```python
fe = smf.ols("purchase ~ mkt_costs + C(city)", data=toy_panel).fit()

fe_toy = toy_panel.assign(y_hat = fe.fittedvalues)

plt.scatter(toy_panel.mkt_costs, toy_panel.purchase, c=toy_panel.city)
for city in fe_toy["city"].unique():
    plot_df = fe_toy.query(f"city=='{city}'")
    plt.plot(plot_df.mkt_costs, plot_df.y_hat, c="C5")

plt.title("Fixed Effect Model")
plt.xlabel("Marketing Costs (in 1000)")
plt.ylabel("In-app Purchase (in 1000)");
```
<img src="https://github.com/DoyoungKim12/causal-inference/blob/master/img_BnT/bnt_45.PNG?raw=true">

- 저 위의 이미지가 수정된 효과모델이 무엇을 하는지에 대해 말해주는 것에 감사하자. 수정된 효과(fixed effect)는 **도시 하나당 하나의 회귀식**을 세우는 것이다. 또한 각 회귀선이 평행하다는 것도 확인하자. 각 회귀선의 기울기는 인앱 구매에 대한 마케팅 비용의 효과를 의미한다. 따라서 **수정된 효과는 인과효과가 모든 개인에 대해 동일하다고 가정한다.** 이는 관점에 따라 약점이 될 수도, 강점이 될 수도 있다. 만약 각 도시별 인과효과를 찾고자 한다면 이는 약점이 된다. 각 도시별 인과효과의 차이를 알 수 없기 때문이다. 그러나 인앱 구매에 대한 마케팅의 전반적인 효과를 찾고자 한다면 이러한 패널 구조의 데이터는 매우 유용한 레버리지가 된다.

<br><br>

## Time Effects
- 




