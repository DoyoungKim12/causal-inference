
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

- (추가 설명) Uplift 모델링에 대한 이해
  - 일반적으로 우리가 알고있는 모델링은 outcome을 예측하는 것으로, 이는 (구매여부를 예측한다고 한다면) P(buy|treatment)를 예측하는 것과 같다.
    - "어떠한 고객이 캠페인을 통한 매상이 높을 것인가?"
  - 그러나, Uplift는 여기에서 한발 더 나아간다. Uplift = P(buy|treatment) - P(buy|not_treatment)로, 아래와 같은 질문에 답할 수 있다.
    - "캠페인이 고객에게 실제 우리회사 제품의 구매를 유발했나?"
    - "이미 사려고 했던 사람에게 캠페인을 하는 낭비 를 하지는 않았나?"
    - "캠페인이 누군가의 구매를 더욱 악화 시키지는 않았나?"

  <br>
  
  - 이제 Uplift tree는 Uplift를 극대화하는 분할 기준을 찾아 split한다. 이때, 기존의 outcome을 예측하는 tree model과의 가장 큰 차이점은 단순히 outcome class의 분포를 보는 것이 아니라 실험군과 대조군의 outcome class 분포 차이를 측정한다는 점이다.
    - 아래 그림을 보면, 분할된 왼쪽 그룹은 T=1일 때 구매/T=0일 때 구매하지 않는 Compliers이고 오른쪽 그룹은 T=1일때 구매하지않음/T=0일때 구매하는 Defiers이다.
    - 즉, 특정 기준으로 Compliers그룹과 Defiers 그룹을 구분해나가는 Tree라고 볼 수 있다.
    - 이제 특정 유닛이 특정 처리군에 속할 때와 대조군에 속할 때의 구매 확률을 각각 구하고, 그 차이로 유닛/처리별 Uplift(ITE의 차이)를 구할 수 있다.  
  
<br>
  
  <img src="https://github.com/DoyoungKim12/causal-inference/blob/master/img_BnT/causalml_1.PNG?" height='400'>
  
<br>

- Uplift 모델링 관련 글이 잘 정리된 블로그 (정리된 4개 포스팅 모두 읽어보면 도움이 될 것이다) : https://jaysung00.github.io/2020/12/17/UM-overview/ 

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
  - 도구변수와 2SLS에 대한 보다 자세한 설명 : https://dodonam.tistory.com/227

<br>

### Targeted maximum likelihood estimation (TMLE) for ATE
- 자동으로(Automated), 이중으로 강건한(Doubly-Robust) 편향되지 않은 ATE 추정치를 계산하는 방법
  - Propensity score, Predicted outcome(Treatment가 0,1인 경우 모두)을 생성한 후, 확률값처럼 0~1 사이의 값으로 min-max scaling된 Predicted outcome으로 Q를 정의한다.
    - Q는 log(p/1-p)의 형태로, 여기서 p는 scaling된 Predicted outcome이다. p는 처리여부에 따라 p0, p1이 각각 존재한다.
  - 다양한 파라미터로 정의된 복잡한 pseudo log-likelihood function를 극대화하는 Q를 구한다.
  - 목적함수를 극대화하는 Q와 다른 파라미터로 Q1과 Q0를 정의, 두 값의 차이로 ATE를 추정한다.

<br>

- Doubly-Robust라는 표현이 쓰이는 것으로 보아, 위에서 보았던 DR ATE 추정치의 발전된 형태로 보인다.
- TMLE에 대한 가벼운 설명 : https://towardsdatascience.com/targeted-maximum-likelihood-tmle-for-causal-inference-1be88542a749
- MLE의 쉽고 빠른 이해 : https://angeloyeo.github.io/2020/07/17/MLE.html


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
    - uplift_trees_with_synthetic_data 
      - https://colab.research.google.com/drive/1RNKzu8N5HmkjrX7peKvwO1KOdHcP3Xrv#scrollTo=KpOBGRK1Yen-
      - Uplift modeling에서는 가능한 모든 treatment에 대해 각각의 유닛별 조건부 전환률(ITE)을 구하고, 각 조건부 전환률의 차이인 Uplift(ITE의 차이)가 가장 높은 treatment를 확인할 수 있다.
      - Uplift(ITE의 차이)가 높게 예측된 상위 K명의 처리 그룹에서 실제로 대조군 대비 전환률이 높았는지를 그래프로 표현하여 모델의 성능을 가늠할 수 있다. (AUUC)
      <br> 
    - meta_learners_with_synthetic_data 
      - https://colab.research.google.com/drive/1kr8R5UiyfPJV4FPXuB5Q9Lpy1IQ9-qFb#scrollTo=_4rmlVZ0HRuZ
      - 가상의 데이터로 actual ATE와 예측된 ATE 분포를 시각화하여 비교, 어떤 모델의 성능이 좋은지 상대적으로 가늠할 수 있다.
      - actual ATE를 알 수 없는 현실에서는 AUUC 정도가 유일한 validation 수단
      <br> 
    - Meta-learners (S/T/X/R) with multiple treatment 
      - https://colab.research.google.com/drive/1sz1vzDuMPs_Ft6azuK4dVhVlzD1pv0LK#scrollTo=mzsRq8_Ww3oO
      - multiple treatment라고 해서 별다른 점이 있는 것은 아니다. treatment array에 몇 개의 유니크한 treatment 그룹이 있는지에 따라 각 그룹별로 ATE/CATE를 계산해 리턴할 뿐이다.
        - 여기서의 CATE는 유닛별로 계산된 Treatment Effect로, ITE와 같은 의미로 이해된다.  
      <br> 
    - Comparing meta-learners across simulation setups 
      - https://colab.research.google.com/drive/16v4r5MNoU1CG2G75pnASMMbqk2fMdzha#scrollTo=rm6vo_8-BaMe
      - 레퍼런스가 되는 paper의 benchmark simulation study를 코드로 구현한 내용
      - 각 모델별로 mse를 비교하여 어떤 모델이 다양한 (가상의) 상황에서 일반적으로 좋은 성능을 보이는지 확인할 수 있다.
      <br>   
    - Doubly Robust (DR) learner
      - https://colab.research.google.com/drive/1PykRJ77vPqLi2axzYLQiumuFkjS2Tz8K#scrollTo=eHBhs6fIKOlo
      - Doubly Robust (DR) learner를 사용하여 ITE를 계산, treatment effect model로 어떤 모델을 쓰는지에 따라 성능이 달라지는지 확인할 수 있다.
        - treatment effect model은 독립변수로 DR 추정치를 fitting하는 모델로, 이제 이 모델에 독립변수들을 input으로 넣으면 DR 추정치가 리턴될 것이다.
      - hidden confounder가 존재할 때, DRIV 모델이 보다 안정적으로 편향되지 않은 ITE를 추정하는 것을 확인할 수 있다.
      <br> 
    - TMLE learner
      - https://colab.research.google.com/drive/1nsKbRIkHuUg7MvyHoueeG-yE14tFvqTK#scrollTo=kG7BFo6ykl3Y 
      - propensity score와 predicted outcome으로 doubly-robust한 ATE 추정치를 찾는 방법
        - 이를 pseudo log-likelihood function를 극대화하는 파라미터로 구한다는 점에서 기존의 DR 방법론과 약간의 차이가 있는 듯 하다. (기존의 DR 방법론은 딱히 뭘 극대화하거나 하지 않고 단순히 가중치를 조정하고 demean을 해주는 정도의 트릭이 들어갔다)
      - 이 예제에서는 ground truth를 대체하는 용도로써 TMLE를 소개한다. ground truth가 존재할 때와 존재하지 않을 때의 gain을 구하는 방법이 각각 다르기 때문에, ground truth가 없을 때의 gain이 자칫 의미없는 것처럼 보일 수 있을 때 TMLE를 사용하여 AUUC를 계산할 수 있다는 것이다.
        - 여기서의 TMLE로 구한 ATE는 사전에 지정한 그룹 수만큼 묶여 grouped gain이 계산된다.
        - input으로는 모든 공변량과 outcome, propensity score, treatment, 그리고 예측된 ITE를 넣어준다. 계산된 gain을 table 또는 plot으로 확인할 수 있다.
        - AUUC 뿐만 아니라 Qini도 쉽게 계산할 수 있다.

<br> 

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

<br><br>

### More algorithms
- Treatment optimization algorithms
  - 아직 실험적인 단계에 있는 방법론이므로 다루지 않음 
- Instrumental variables algorithms
  - 2SLS method
    - https://colab.research.google.com/drive/13x2I5tC0_xz3AzSTujil-Vmp67LuZbq3#scrollTo=nWzfFOtoCRPF
    - OLS에 비해 true value에 가까운 ATE를 리턴하는 것을 확인할 수 있다.  
- Neural network based algorithm 
  - Deep Model에 대해서는 다루지 않음 
- Validation
  - AUUC 외의 것에 대해서는 다루지 않음  

<br><br>

### Interpretable Causal ML 
- https://colab.research.google.com/drive/1QT34BN0Usx4_jLuocwHMuyIVqCZKgyqb#scrollTo=FavHa2YZ5e-I

<br>

- Meta-Learner Feature Importances
  - Meta-Learner의 각 피쳐별 중요도를 확인할 수 있다. (get_importance)
  -  Currently supported methods are:
      - auto (calculates importance based on estimator's default implementation of feature importance; estimator must be tree-based) <br> Note: if none provided, it uses lightgbm's LGBMRegressor as estimator, and "gain" as importance type
      - permutation (calculates importance based on mean decrease in accuracy when a feature column is permuted; estimator can be any form) <br> Hint: for permutation, downsample data for better performance especially if X.shape[1] is large
  - SHAP Value의 확인도 가능하다. (get_shap_values)
  
<br>

```python
from causalml.inference.meta import BaseSRegressor, BaseTRegressor, BaseXRegressor, BaseRRegressor

slearner = BaseSRegressor(LGBMRegressor(), control_name='control')
slearner.estimate_ate(X, w_multi, y)
slearner_tau = slearner.fit_predict(X, w_multi, y)

model_tau_feature = RandomForestRegressor()  # specify model for model_tau_feature

slearner.get_importance(X=X, tau=slearner_tau, model_tau_feature=model_tau_feature,
                        normalize=True, method='auto', features=feature_names)

# Using the feature_importances_ method in the base learner (LGBMRegressor() in this example)
slearner.plot_importance(X=X, tau=slearner_tau, normalize=True, method='auto')

# Using eli5's PermutationImportance
# 특정 피쳐를 제거했을 때 발생하는 성능의 손실을 계산하는 방법
slearner.plot_importance(X=X, tau=slearner_tau, normalize=True, method='permutation')

# Using SHAP
shap_slearner = slearner.get_shap_values(X=X, tau=slearner_tau)

# Plot shap values without specifying shap_dict
slearner.plot_shap_values(X=X, tau=slearner_tau)

# Plot shap values WITH specifying shap_dict
slearner.plot_shap_values(X=X, shap_dict=shap_slearner)

# interaction_idx set to 'auto' (searches for feature with greatest approximate interaction)
slearner.plot_shap_dependence(treatment_group='treatment_A',
                            feature_idx=1,
                            X=X,
                            tau=slearner_tau,
                            interaction_idx='auto')
```

<img src="https://github.com/DoyoungKim12/causal-inference/blob/master/img_BnT/causalml_2.PNG?" height='400'>
<img src="https://github.com/DoyoungKim12/causal-inference/blob/master/img_BnT/causalml_3.PNG?" height='400'>
<img src="https://github.com/DoyoungKim12/causal-inference/blob/master/img_BnT/causalml_4.PNG?" height='400'>


<br>

- Uplift Tree Visualization 
  - Tree가 split하는 형태를 graphviz로 시각화하여 보여준다.
```python
from IPython.display import Image
from causalml.inference.tree import UpliftTreeClassifier, UpliftRandomForestClassifier
from causalml.inference.tree import uplift_tree_string, uplift_tree_plot
from causalml.dataset import make_uplift_classification

df, x_names = make_uplift_classification()
uplift_model = UpliftTreeClassifier(max_depth=5, min_samples_leaf=200, min_samples_treatment=50,
                                    n_reg=100, evaluationFunction='KL', control_name='control')

uplift_model.fit(df[x_names].values,
                treatment=df['treatment_group_key'].values,
                y=df['conversion'].values)

graph = uplift_tree_plot(uplift_model.fitted_uplift_tree, x_names)
Image(graph.create_png())
```

<img src="https://github.com/DoyoungKim12/causal-inference/blob/master/img_BnT/causalml_5.PNG?" height='400'>

<br>

- Uplift Tree Feature Importances
```python
pd.Series(uplift_model.feature_importances_, index=x_names).sort_values().plot(kind='barh', figsize=(12,8))
```

<img src="https://github.com/DoyoungKim12/causal-inference/blob/master/img_BnT/causalml_6.PNG?" height='400'>

<br><br>

### Synthetic Data Generation Process
- 각각 Meta learner 실습용 notebook에서 확인할 수 있음

<br><br>

### Sensitivity Analysis

<br><br>

### Feature Selection
- https://colab.research.google.com/drive/1CDNccz8ctkMesHOhzk3yCdShpW73fFk5#scrollTo=OospaFuxrJc3
- 일반적인 변수선택법은 outcome과 feature의 상관관계를 관찰하여 결정하는데, 우리가 만드는 모델의 타겟은 outcome이 아닌 uplift이므로 이는 최선이 아닐 수 있다.
- 따라서 Uplift Modeling에 최적화된 Feature Selection 방법론을 아래와 같이 제시한다.
  - Filter Methods
    - 처리 여부를 나타내는 변수 Z와 확인하고자하는 feature, 그리고 그들의 교호작용항(interaction term)을 사용하여 outcome변수를 예측하는 선형회귀모델을 구성한다.
    - 교호작용항의 계수에만 F 검정을 시행, 이 통계값이 크면 해당 feature는 강한 heterogeneous treatment effect와 상관이 있다는 것을 의미한다.
  - LR filter (Likelihood ratio)
    - 로지스틱 회귀모형의 교호작용항 계수에 대한 likelihood ratio 검정 통계량으로 정의한다.
  - Filter method with K bins
    - 샘플을 feature의 백분위를 기준으로 K개의 bin으로 나눈다. (여기서 K는 하이퍼파라미터) 
    - 중요도 점수는 이러한 K개의 bins에 대한 처리효과의 divergence measure로 정의된다. (measure의 종류는 uplift tree의 split criterion과 같다)

<br>

- 변수선택법에 대한 보다 자세한 정리 : https://jaysung00.github.io/2020/12/17/Selection/

```python
from causalml.feature_selection.filters import FilterSelect
from causalml.dataset import make_uplift_classification

# define parameters for simulation
y_name = 'conversion'
treatment_group_keys = ['control', 'treatment1']
n = 100000
n_classification_features = 50
n_classification_informative = 10
n_classification_repeated = 0
n_uplift_increase_dict = {'treatment1': 8}
n_uplift_decrease_dict = {'treatment1': 4}
delta_uplift_increase_dict = {'treatment1': 0.1}
delta_uplift_decrease_dict = {'treatment1': -0.1}

# make a synthetic uplift data set
random_seed = 20200808
df, X_names = make_uplift_classification(
    treatment_name=treatment_group_keys,
    y_name=y_name,
    n_samples=n,
    n_classification_features=n_classification_features,
    n_classification_informative=n_classification_informative,
    n_classification_repeated=n_classification_repeated,
    n_uplift_increase_dict=n_uplift_increase_dict,
    n_uplift_decrease_dict=n_uplift_decrease_dict,
    delta_uplift_increase_dict = delta_uplift_increase_dict,
    delta_uplift_decrease_dict = delta_uplift_decrease_dict,
    random_seed=random_seed
)

# Feature selection with Filter method
filter_f = FilterSelect()
method = 'F'
f_imp = filter_f.get_importance(df, X_names, y_name, method,
                      treatment_group = 'treatment1')
print(f_imp)

# Use likelihood ratio test method
method = 'LR'
lr_imp = filter_f.get_importance(df, X_names, y_name, method,
                      treatment_group = 'treatment1')
print(lr_imp)

# Use KL divergence method
method = 'KL'
kl_imp = filter_f.get_importance(df, X_names, y_name, method,
                      treatment_group = 'treatment1',
                      n_bins=10)
print(kl_imp)
```
