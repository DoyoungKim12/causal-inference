
<br><br>
# CausalML
https://causalml.readthedocs.io/en/latest/about.html <br>
**인과추론**과 **uplift 모델링**을 위한 다양한 도구를 제공하는 파이썬 패키지인 CausalML에 대해 알아보자 

<br><br>
## About CausalML
### CausalML이란?
- 최근의 연구에 기반하여 머신러닝 알고리즘을 사용한 Uplift Modeling과 Causal Inference 도구를 제공하는 파이썬 패키지
- CATE(Conditional Average Treatment Effect) 또는 ITE(Individual Treatment Effect)를 실험 또는 관찰데이터로부터 추정할 수 있는 표준 인터페이스 제공
- 기본적으로, 각 개인에 대하여 관찰된 피쳐 X로 개입(처리) T가 결과 Y에 미치는 인과효과를 추정하며 모델 형태에 대한 강한 가정을 두지 않음

<br>

### 일반적인 사용 예시
- 캠페인 타겟팅 최적화 : 광고 캠페인에서 ROI를 높이기 위한 중요한 포인트는 호의적인 반응을 얻을 수 있는 고객을 대상으로 광고를 공략하는 것으로, CATE는 A/B test 또는 과거 관측 데이터로부터 개별 수준의 광고 노출로 인한 KPI의 효과를 추정하여 고객군 식별
- 맞춤형 Engagement : 고객과 상호작용할 수 있는 여러가지 옵션이 존재할 때, CATE를 활용하여 고객별 처리효과와 최적화된 처리 옵션을 추정

<br>

### 패키지가 제공하는 Method 목록
- Tree-based algorithms
  - Uplift Random Forests on KL divergence, Euclidean Distance, and Chi-Square
  - Uplift Random Forests on Contextual Treatment Selection
  - Uplift Random Forests on delta-delta-p criterion (only for binary trees and two-class problems)
- Meta-learner algorithms
  - S-Learner
  - T-Learner
  - X-Learner
  - R-Learner
  - Doubly Robust (DR) learner
- Instrumental variables algorithms
  - 2-Stage Least Squares (2SLS)
  - Doubly Robust Instrumental Variable (DRIV) learner
- Neural network based algorithms
  - CEVAE
  - DragonNet
- Treatment optimization algorithms
  - Counterfactual Unit Selection
  - Counterfactual Value Estimator

<br><br>

## Methodology
### Meta-Learner Algorithms
- Meta-Learner는 (base learners로 불리는) 머신러닝 알고리즘을 사용하여 CATE를 추정하는 것으로, 머신러닝 알고리즘은 어떤 것이라도 사용될 수 있다. 처리 여부를 피쳐로 사용하는 Single base learner를 사용하거나 각 처리군과 대조군 별로 여러개의 base learner를 사용할 수도 있다.

<br>

- S-learner, T-learner, X-learner는 최근에 다루었으므로 설명 생략
- R-learner
  - The R-learner provides a general framework to estimate the heterogeneous treatment effect tau(X). We first estimate **marginal effects** and **treatment propensities** in order to form an objective function that isolates the causal component of the signal. Then, we optimize this **data-adaptive objective function**. The R-learner is flexible and easy to use: For both steps, we can use any loss-minimization method, e.g., the lasso, random forests, boosting, etc.; moreover, these methods can be fine-tuned by cross validation.

<br>

- Doubly Robust (DR) learner

Doubly Robust Estimation 복습 : propensity score와 outcome에 대한 선형회귀를 조합하여 ATE를 추정하는 방법으로, propensity score 모델과 outcome 모델 중 하나라도 제대로 추정되면 되기 때문에 Doubly Robust라는 이름이 붙었다. 아래 코드로 간단하게 구현할 수 있다.

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
<br>

CausalML에서는 이를 3-fold CV 추정치를 사용하는 방식으로 구현하였다. 일단 데이터를 3개의 서브그룹으로 나누고 첫번째 그룹으로 propensity score 모델을 학습, 두번재 그룹으로 potential outcome 모델을 학습시킨다. 학습된 두 개의 모델로 세번째 그룹의 추정치를 반환하는데, 이는 위의 파이썬 코드에서 리턴하는 값과 정확히 같다. 이를 각 Stage마다 사용되는 서브그룹을 교체하는 방식으로 3회 수행하여, 3개 CATE의 평균을 반환한다.

<br>

- Doubly Robust Instrumental Variable (DRIV) learner
  - DR-learner에 도구변수 Z(assignment status)를 추가하여 LATE를 추정하는 모델로, LATE는 Defier(청개구리) 그룹을 제외한 Compliers 그룹에 대한 인과효과를 의미한다. 도구변수 Z의 역할은 처리 T를 outcome처럼 취급하여 Z에 따른 T를 추정하는 것이다. 여기서의 가정은 Z=1일 때의 T가 Z=0일 때의 T보다 커야한다는 것으로, 이는 곧 Compliers의 정의와 같다.
  - DR-learner에서와 같이 3-fold CV 추정치를 사용하는 방식으로 구현되었다. 

<br><br>

### Tree-Based Algorithms
- Uplift Tree
  - Uplift Tree 접근법은 분할(splitting) 기준이 Uplift의 차이에 기초하는 트리 기반 알고리즘을 사용하는 일련의 방법으로 구성된다. gain을 축정하는 3가지 방식을 제시하며, 이는 각각 KL divergence, Euclidean Distance, Chi-square divergence이다.
  - DDP와 CTS도 각각 샘플의 분할 기준 중 하나를 의미한다. 

<br>

### Value optimization methods
- Counterfactual Unit Selection, Counterfactual Value Estimator
  - 유저를 4개 범주로 분류, 각 그룹별로 얻는 이익의 합을 극대화하는 방법
  - N.B. The current implementation in the package is highly experimental

<br>

### Selected traditional methods
- Matching
  - RCT를 흉내내기 위해 실험군과 대조군에서 서로 가장 비슷한 유닛끼리 매칭하는 방법 
- Inverse probability of treatment weighting
  - IPTW 접근방식은 실험군과 대조군이 관찰된 피쳐의 관점에서 유사한 인공 모집단을 생성하는 것 (propensity score의 역수를 가중치로 사용)
  - 단순 매칭과 비교했을 때 IPTW의 장점은 처리군과 대조군의 유사성 부족으로 인해 폐기되는 데이터가 더 적을 수 있다는 것이다. 단점으로는 극단적인 성향 점수가 매우 가변적인 추정치를 생성할 수 있다.
- 2-Stage Least Squares (2SLS)
  - 도구변수는 OLS의 5개 가정 중 내생성을 충족시키기 위한 트릭이다. 내생성은 제 3의 요인이 독립변수 X와 종속변수 Y에 동시에 영향을 미칠 때 발생한다. 이러한 교란을 일으키는 요인을 U라고 할 때, U와는 낮은 상관성을 가지면서 독립변수 X와 높은 상관성을 가지는 도구변수 Z를 찾을 수 있다. 이 Z로 X를 회귀하는 회귀식을 구성하고(First stage), 이를 이용해 추정된 결과를 사용하여 Y를 추정(Second stage)하면 U로 인해 발생하는 외생성을 제거할 수 있다. (U와의 낮은 상관성은 error term으로 확인할 수 있다. error term과의 상관성이 낮으면 교란을 일으키는 무언가와의 상관성이 낮은 것으로 볼 수 있다.) 
  - https://dodonam.tistory.com/227

<br><br>

## Installation
- conda를 사용하여 설치하는 방법을 권장
- pip를 사용하여 아래와 같이 설치할 수 있음
  - requirements에는 xgboost, lightgbm과 같은 트리 기반 부스팅 모델과 딥모델을 위한 Pytorch(torch)가 포함되어있음
  - 데이터브릭스에서도 클러스터에 requirements만 잘 설치하면 사용하는 데에는 크게 무리가 없어보임
  
```python
$ git clone https://github.com/uber/causalml.git
$ cd causalml
$ pip install -r requirements.txt
$ pip install causalml
```

<br><br>

## Examples

### Propensity Score
- Propensity Score Estimation
  - Propensity Score를 추정할 모델을 골라서 fit_predict

```python
# GradientBoostedPropensityModel, LogisticRegressionPropensityModel도 있음
from causalml.propensity import ElasticNetPropensityModel

pm = ElasticNetPropensityModel(n_fold=5, random_state=42)
ps = pm.fit_predict(X, y)
```

<br>

- Propensity Score Matching
  - 매칭결과를 데이터프레임으로 반환한다. Propensity Score를 추정하는 모델이나 유사도를 계산하는 부분에 대해서는 설명되지 않았다.

```python
from causalml.match import NearestNeighborMatch, create_table_one

# psm은 아마 Propensity Score Matching의 줄임말일 것
psm = NearestNeighborMatch(caliper=0.2, # 매칭의 기준이 되는 threshold
                           replace=False, # 매칭시 복원추출을 사용할 것인지의 여부 
                           ratio=1, # 실험군과 대조군의 비율
                           random_state=42
                          )
                           
# 매칭 결과를 pandas dataframe으로 return                           
matched = psm.match_by_group(data=df,
                             treatment_col=treatment_col,
                             score_col=score_col,
                             groupby_col=groupby_col # 계층화(stratification)가 필요할 경우 사용
                            )

# 매칭 결과에 대한 요약 리포트를 pandas dataframe으로 return
# 실험군과 대조군의 각 feature별 평균, 표준편차, standard mean difference (SMD)를 계산
create_table_one(data=matched,
                 treatment_col=treatment_col,
                 features=covariates)
```

<br><br>

### Average Treatment Effect (ATE) Estimation
- Meta-learners and Uplift Trees
  - 각각 다른 Meta-learner라도 같은 method인 estimate_ate로 쉽게 ATE를 추정할 수 있다.
  - Example notebook을 참고
    - https://colab.research.google.com/drive/1RNKzu8N5HmkjrX7peKvwO1KOdHcP3Xrv#scrollTo=KpOBGRK1Yen-
    - https://colab.research.google.com/drive/1kr8R5UiyfPJV4FPXuB5Q9Lpy1IQ9-qFb#scrollTo=_4rmlVZ0HRuZ

```python
from causalml.inference.meta import LRSRegressor
from causalml.inference.meta import XGBTRegressor, MLPTRegressor
from causalml.inference.meta import BaseXRegressor
from causalml.inference.meta import BaseRRegressor
from xgboost import XGBRegressor
from causalml.dataset import synthetic_data

y, X, treatment, _, _, e = synthetic_data(mode=1, n=1000, p=5, sigma=1.0)

lr = LRSRegressor()
te, lb, ub = lr.estimate_ate(X, treatment, y)
print('Average Treatment Effect (Linear Regression): {:.2f} ({:.2f}, {:.2f})'.format(te[0], lb[0], ub[0]))

xg = XGBTRegressor(random_state=42)
te, lb, ub = xg.estimate_ate(X, treatment, y)
print('Average Treatment Effect (XGBoost): {:.2f} ({:.2f}, {:.2f})'.format(te[0], lb[0], ub[0]))

nn = MLPTRegressor(hidden_layer_sizes=(10, 10),
                 learning_rate_init=.1,
                 early_stopping=True,
                 random_state=42)
te, lb, ub = nn.estimate_ate(X, treatment, y)
print('Average Treatment Effect (Neural Network (MLP)): {:.2f} ({:.2f}, {:.2f})'.format(te[0], lb[0], ub[0]))

xl = BaseXRegressor(learner=XGBRegressor(random_state=42))
te, lb, ub = xl.estimate_ate(X, treatment, y, e)
print('Average Treatment Effect (BaseXRegressor using XGBoost): {:.2f} ({:.2f}, {:.2f})'.format(te[0], lb[0], ub[0]))

rl = BaseRRegressor(learner=XGBRegressor(random_state=42))
te, lb, ub =  rl.estimate_ate(X=X, p=e, treatment=treatment, y=y)
print('Average Treatment Effect (BaseRRegressor using XGBoost): {:.2f} ({:.2f}, {:.2f})'.format(te[0], lb[0], ub[0]))
```






