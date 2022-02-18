
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


















