<br><br><br><br>

# 12 - Doubly Robust Estimation

<br><br>

## Don’t Put All your Eggs in One Basket
- 우리는 지난 시간에 $E\[Y\barT=1] - E\[Y\barT=0]\bar X$를 추정할 때 선형회귀와 propensity score weighting을 어떻게 사용하는지 배웠다. 
- 그런데 어떤 것을 언제 사용해야 되는지는 아직 모른다. 이럴 때에는, 그냥 **둘 다 사용하면 된다!** Doubly Robust Estimation은 propensity score와 선형회귀를 조합한 방식으로, 이제 우리는 둘 중 하나에만 의존하지 않아도 된다.
<br><br>

- 이게 어떻게 동작하는지 알아보기 위해, 마인드셋 실험을 다시 가져와보자. 
  - 마인드셋 실험 : 미국의 여러 공립 고등학교에서 수행된 무작위 실험으로, growth mindset의 효과를 찾는 것을 목적으로 한다. 이 실험은 학교에서 세미나 수업을 들은 학생들이 내면에 growth mindset을 갖게 되는 방식으로 작동한다. 이후, 해당 학생들의 대학 기록을 추적하여 그들이 학문적으로 어느 정도의 성취를 이루었는지 측정한다. 측정된 값은 표준화되었다. ( 
  - 실제 데이터는 사생활 보호 차원에서 공개되지 않았으나, 같은 통계적 분포를 보이는 시뮬레이션 데이터를 사용할 수 있다. 

```python
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from matplotlib import style
from matplotlib import pyplot as plt
import seaborn as sns

%matplotlib inline

style.use("fivethirtyeight")
pd.set_option("display.max_columns", 6)

data = pd.read_csv("./data/learning_mindset.csv")
data.sample(5, random_state=5)
```
<img src="https://github.com/DoyoungKim12/causal-inference/blob/master/img_BnT/bnt_25.PNG?raw=true">

<br><br>

- 무작위 실험이라고 할지라도, 이 데이터가 교란으로부터 자유롭지는 않아보인다. 세미나에 참가할 기회는 무작위로 제공되었으나, 참가를 결정하는 것은 개인의 의지로 무작위가 아니었기 때문이다. 말하자면 우리는 지금 여기서 non-compliance의 사례를 다루고 있는 것이다. (처리 여부에 상관없이 무조건 처리를 받지 않는 경우) 이에 대한 하나의 증거는 학생들의 성공에 대한 기대치가 세미나 참석여부와 상관관계를 보인다는 점이다. 더 높은 성공 기대치를 가진 학생 그룹은 세미나에 참가하는 비율도 높았다. 
```python
data.groupby("success_expect")["intervention"].mean()
```
<img src="https://github.com/DoyoungKim12/causal-inference/blob/master/img_BnT/bnt_26.PNG?raw=true">

<br><br>

- 지금까지 우리가 아는 바로는, 여기에 선형회귀를 사용하거나 로지스틱 회귀로 propensity score를 추정하는 것을 적용할 수 있었다. 이러한 것들을 하기 전에 일단 범주형 변수를 더미 변수로 변환해주자. 이제 우리는 doubly robust estimation이 어떻게 동작하는지 이해할 준비가 되었다.
```python
categ = ["ethnicity", "gender", "school_urbanicity"]
cont = ["school_mindset", "school_achievement", "school_ethnic_minority", "school_poverty", "school_size"]

data_with_categ = pd.concat([
    data.drop(columns=categ), # dataset without the categorical features
    pd.get_dummies(data[categ], columns=categ, drop_first=False) # categorical features converted to dummies
], axis=1)

print(data_with_categ.shape)
# (10391, 32)
```

<br><br>

## doubly robust estimation

<img src="https://github.com/DoyoungKim12/causal-inference/blob/master/img_BnT/bnt_27.PNG?raw=true">

- 이 추정기(estimator)를 도출해내기 전에, 먼저 이걸 보여주고 이게 왜 대단한지에 대해서만 이야기할 것이다. 

<img src="https://github.com/DoyoungKim12/causal-inference/blob/master/img_BnT/bnt_28.PNG?raw=true">
- $\hat{P(x)}$ : propensity score 추정치 (로지스틱 회귀를 사용했다고 가정하자.)
- $\hat{\mu_1(x)} : $E\[Y\barX,T=1]$의 추정치
- $\hat{\mu_0(x)} : $E\[Y\barX,T=0]$의 추정치
- 눈치챘겠지만, 우변에서의 첫번째 항이 $E\[Y_1]$의 추정치이고 두번재 항이 $E\[Y_0]$의 추정치이다. 일단 첫번째 항을 살펴보자. 그럼 두번째 항도 같은 직관이 적용되므로 유추할 수 있을 것이다. 

<br><br>

- 위의 공식이 처음에는 무서워보인다는 것을 알기 때문에, (하지만 걱정하지 말자. 나중에는 이게 엄청 간단하다는 것을 알게 될것이다.) 일단 이 추정기를 어떻게 코드로 표현하는지를 먼저 보여줄것이다. 어떤 사람들은 코드를 덜 두려워하는 감이 있기도 하다. 실제로 이 추정기가 어떻게 작동하는지 함께 살펴보자.
```python
from sklearn.linear_model import LogisticRegression, LinearRegression

def doubly_robust(df, X, T, Y):
    ps = LogisticRegression(C=1e6).fit(df[X], df[T]).predict_proba(df[X])[:, 1]
    mu0 = LinearRegression().fit(df.query(f"{T}==0")[X], df.query(f"{T}==0")[Y]).predict(df[X]) # mu0는 untreated로 train하여 모든 개인의 untreated 상태의 outcome을 추정하는 것
    mu1 = LinearRegression().fit(df.query(f"{T}==1")[X], df.query(f"{T}==1")[Y]).predict(df[X]) # mu1는 treated로 train하여 모든 개인의 treated 상태의 outcome을 추정하는 것
    return (
        np.mean(df[T]*(df[Y] - mu1)/ps + mu1) -
        np.mean((1-df[T])*(df[Y] - mu0)/(1-ps) + mu0)
    )
```
```python
T = 'intervention'
Y = 'achievement_score'
X = data_with_categ.columns.drop(['schoolid', T, Y])

doubly_robust(data_with_categ, X, T, Y)
# 0.38822192386353527
```

<br><br>

- Doubly robust estimator는 마인드셋 세미나에 참석한 개인이 참석하지 않은 사람들에 비해 0.388 표준편차만큼 더 높은 성취를 보였다고 말하고 있다. 다시 한 번, 우리는 신뢰구간 추정을 위해 부트스트랩을 이용할 수 있다. 
```python
from joblib import Parallel, delayed # for parallel processing

np.random.seed(88)
# run 1000 bootstrap samples
bootstrap_sample = 1000
ates = Parallel(n_jobs=4)(delayed(doubly_robust)(data_with_categ.sample(frac=1, replace=True), X, T, Y)
                          for _ in range(bootstrap_sample))
ates = np.array(ates)
```
```python
print(f"ATE 95% CI:", (np.percentile(ates, 2.5), np.percentile(ates, 97.5)))
# ATE 95% CI: (0.3536507259630512, 0.4197834129772669)
```
```python
sns.distplot(ates, kde=False)
plt.vlines(np.percentile(ates, 2.5), 0, 20, linestyles="dotted")
plt.vlines(np.percentile(ates, 97.5), 0, 20, linestyles="dotted", label="95% CI")
plt.title("ATE Bootstrap Distribution")
plt.legend();
```
<img src="https://github.com/DoyoungKim12/causal-inference/blob/master/img_BnT/bnt_29.PNG?raw=true">

<br><br>

- doubly robust estimator를 이렇게 '찍먹'해보았으니, 왜 이 방법론이 위대한 것인지 보다 면밀히 들여다보자. 먼저 이 방법론이 doubly robust라고 불리는 이유는 $\hat{P(x)}$ 또는 $\hat{\mu(x)}$의 두 모델 중 하나가 올바르게 특정되는 것만을 요구하기 때문이다. 이걸 이해하기 위해, 첫번째 항을 좀 더 자세히 살펴보자.
<img src="https://github.com/DoyoungKim12/causal-inference/blob/master/img_BnT/bnt_30.PNG?raw=true">
- $\hat{\mu_1(x)}$가 올바른 추정치라고 가정해보자. 만약 propensity score가 틀렸다고 하더라도, 우리는 걱정하지 않아도 된다. 왜냐하면 $\hat{\mu_1(x)}$가 올바른 추정치라면 $E\[T_i(Y_i-\hat{\mu_1(X_i)})] = 0$이 되기 때문이다. 
  - $T_i$를 곱하는 것은 (여기서는 i가 1이기 때문에) 처리를 받은 집단만을 선택하는 것이고 처리군에 대한 $\hat{\mu_1}$의 잔차는 (정의한 바에 따르면) 0이기 때문이다.
  - 이는 모든 것들이 $\hat{\mu_1(X_i)}$로 축약되는 결과를 가져오고, 이는 가정에 의해 $E\[Y_1]$의 올바른 추정치가 된다. 
  - 따라서 (올바르다고 가정한) $\hat{\mu_1(X_i)}$은 propensity score model의 관련성을 제거한다. 그리고 이를 두번째 항에 그대로 적용할 수 있다.

<br><br>

- 하지만 내 말을 그대로 믿지는 말라. 말이 아닌 코드가 증명할 것이다. 아래의 추정기에서, 나는 propensity score를 추정하는 로지스틱 회귀 모델을 0.1과 0.9 사이의 값을 갖는 random uniform variable (모든 경우의 확률이 같은 분포에서 추출되는 무작위 값)로 바꿔두었다. (나는 내 propensity score variance를 망치는 아주 작은 가중치를 원하지 않는다.) 값이 랜덤이기 때문에, 당연히 이는 좋은 propensity score이 될 리가 없다. 그러나 우리는 doubly robust estimator가 로지스틱 회귀로 추정한 propensity score가 주어졌을 때와 매우 근사한 추정치를 생성하려고 하는 모습을 관찰할 수 있다. 
```python
from sklearn.linear_model import LogisticRegression, LinearRegression

def doubly_robust_wrong_ps(df, X, T, Y):
    # wrong PS model
    np.random.seed(654)
    ps = np.random.uniform(0.1, 0.9, df.shape[0])
    mu0 = LinearRegression().fit(df.query(f"{T}==0")[X], df.query(f"{T}==0")[Y]).predict(df[X])
    mu1 = LinearRegression().fit(df.query(f"{T}==1")[X], df.query(f"{T}==1")[Y]).predict(df[X])
    return (
        np.mean(df[T]*(df[Y] - mu1)/ps + mu1) -
        np.mean((1-df[T])*(df[Y] - mu0)/(1-ps) + mu0)
    )
```
```python
doubly_robust_wrong_ps(data_with_categ, X, T, Y)
# 0.37982453125218174
```

- 부트스트랩을 사용해보면, 분산이 제대로 추정된 propensity score를 사용했을 때보다 약간 높아진 것을 알 수 있다. (추정치의 불확실성이 증가함)
```python
np.random.seed(88)
parallel_fn = delayed(doubly_robust_wrong_ps)
wrong_ps = Parallel(n_jobs=4)(parallel_fn(data_with_categ.sample(frac=1, replace=True), X, T, Y)
                              for _ in range(bootstrap_sample))
wrong_ps = np.array(wrong_ps)
```
```python
print(f"ATE 95% CI:", (np.percentile(ates, 2.5), np.percentile(ates, 97.5)))
# ATE 95% CI: (0.3536507259630512, 0.4197834129772669)
```

<br><br>

- 이는 propensity score 모델이 잘못되었지만 outcome 모델이 올바르게 추정하는 경우를 커버한다. 그런데 다른 상황은 또 어떨까? 첫번째 항을 다시 한번 잘 살펴보되, 식의 특정 term을 조금 다르게 풀어서 써보자.

<img src="https://github.com/DoyoungKim12/causal-inference/blob/master/img_BnT/bnt_31.PNG?raw=true">

- 










