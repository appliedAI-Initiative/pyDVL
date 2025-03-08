---
title: Semi-values
alias:
  name: semi-values-intro
---

# Semi-values

Shapley Value is a particular case of a more general concept called semi-value,
which is a generalization to different weighting schemes. A **semi-value** is
any valuation function with the form:

$$
v_\text{semi}(i) = \sum_{i=1}^n w(k)
\sum_{S \sim P(D_{-i}^{(k)})} [u(S_{+i}) - u(S)],
$$

where the coefficients $w(k)$ satisfy the property:

$$\sum_{k=1}^n w(k) = 1,$$

and $P(D_{-i}^{(k)})$ is a distribution over all subsets of $D$ of size $k$ that
do not include sample $x_i$, $S_{+i}$ is the set $S$ with $x_i$ added, and $u$
is the utility function.

Two particular instances of this are **Banzhaf indices** [@wang_data_2023] and
**Beta Shapley** [@kwon_beta_2022], which offer improved numerical and rank
stability in certain situations. See below for more details.

All semi-values, including those two, are implemented in pyDVL by composing
different sampling methods and weighting schemes. The abstract class from which
they derive is
[SemiValueValuation][pydvl.valuation.methods.semivalue.SemivalueValuation],
whose main abstract method is the  weighting scheme $k \mapsto w(k)$.


## Beta Shapley

For some machine learning applications, where the utility is typically the
performance when trained on a set $S \subset D$, diminishing returns are often
observed when computing the marginal utility of adding a new data point.

Beta Shapley is a weighting scheme that uses the Beta function to place more
weight on subsets deemed to be more informative. The weights are defined as:

$$
w(k) := \frac{B(k+\beta, n-k+1+\alpha)}{B(\alpha, \beta)},
$$

where $B$ is the [Beta function](https://en.wikipedia.org/wiki/Beta_function),
and $\alpha$ and $\beta$ are parameters that control the weighting of the
subsets. Setting both to 1 recovers Shapley values, and setting $\alpha = 1$,
and $\beta = 16$ is reported in [@kwon_beta_2022] to be a good choice for some
applications. Beta Shapley values are available in pyDVL through
[BetaShapleyValuation][pydvl.valuation.methods.beta_shapley.BetaShapleyValuation]:

??? Example "Beta Shapley values"
    ```python
    from pydvl.valuation import *
    
    model = ...
    train, test = Dataset.from_arrays(...)
    scorer = SupervisedScorer(model, test, default=0.0)
    utility = ModelUtility(model, scorer)
    sampler = PermutationSampler()
    stopping = RankCorrelation(rtol=1e-5, burn_in=100) | MaxUpdates(2000)
    valuation = BetaShapleyValuation(
        utility, sampler, stopping, alpha=1, beta=16
    )
    valuation.fit(train)
    ```

See, however, the [Banzhaf indices](#banzhaf-indices) section for an alternative
choice of weights which is reported to work better in cases of high variance in
the utility function.


## Banzhaf indices

As noted in the section [Problems of Data Values][problems-of-data-values], the
Shapley value can be very sensitive to variance in the utility function. For
machine learning applications, where the utility is typically the performance
when trained on a set $S \subset D$, this variance is often largest for smaller
subsets $S$. It is therefore reasonable to try reducing the relative
contribution of these subsets with adequate weights.

One such choice of weights is the Banzhaf index, which assigns a constant weight:
 
$$
w(k) := 2^{-(n-1)},
$$

for all set sizes $k$. The intuition for picking a constant weight is that for
any choice of weight function $w$, one can always construct a utility with
higher variance where $w$ is greater. Therefore, in a worst-case sense, the best
one can do is to pick a constant weight.

The authors of [@wang_data_2023] show that Banzhaf indices are more robust to
variance in the utility function than Shapley and Beta Shapley values. They are
available in pyDVL through
[BanzhafValuation][pydvl.valuation.methods.banzhaf.BanzhafValuation]:

```python
from joblib import parallel_config
from pydvl.valuation import (
    ModelUtility, Dataset, SupervisedScorer, PermutationSampler
)
from pydvl.valuation.methods.banzhaf import BanzhafValuation
from pydvl.valuation.stopping import MinUpdates

train, test = Dataset.from_arrays(...)
model = ...
utility = ModelUtility(model, SupervisedScorer(model, test, default=0.0))
sampler = PermutationSampler()
valuation = BanzhafValuation(utility, sampler, MinUpdates(1000))
with parallel_config(n_jobs=16):
    valuation.fit(train)
```

### Banzhaf semi-values with MSR sampling

Wang et al. propose a more sample-efficient method for computing Banzhaf 
semi-values in their paper *Data Banzhaf: A Robust Data Valuation Framework 
for Machine Learning* [@wang_data_2023]. This method updates all semi-values
per each evaluation of the utility (i.e. per model training) based on whether a 
specific data point was included in the data subset or not. The expression 
for computing the semi-values is

$$
\hat{\phi}_{MSR}(i) = \frac{1}{|\mathbf{S}_{\ni i}|} \sum_{S \in 
\mathbf{S}_{\ni i}} U(S) - \frac{1}{|\mathbf{S}_{\not{\ni} i}|} 
\sum_{S \in \mathbf{S}_{\not{\ni} i}} U(S)
$$

where $\mathbf{S}_{\ni i}$ are the subsets that contain the index $i$ and 
$\mathbf{S}_{\not{\ni} i}$ are the subsets not containing the index $i$.

pyDVL provides a sampler for this method, called
[MSRSampler][pydvl.valuation.samplers.msr.MSRSampler], which can be combined
with any valuation method, including
[BanzhafValuation][pydvl.valuation.methods.banzhaf.BanzhafValuation]. However,
because the sampling probabilities induced by MSR coincide with Banzhaf indices,
it is preferred to use the dedicated class
[MSRBanzhafValuation][pydvl.valuation.methods.banzhaf.MSRBanzhafValuation]. For
more on this subject see [[semi-values-sampling]].

??? Example "MSR Banzhaf values"

    ```python
    from joblib import parallel_config
    from pydvl.valuation import ModelUtility, Dataset, SupervisedScorer
    from pydvl.valuation.methods.banzhaf import MSRBanzhafValuation
    from pydvl.valuation.stopping import MaxSamples
    
    train, test = Dataset.from_arrays(...)
    model = ...
    utility = ModelUtility(model, SupervisedScorer(model, test, default=0.0))
    valuation = MSRBanzhafValuation(utility, MaxSamples(1000), batch_size=64)
    with parallel_config(n_jobs=16):
        valuation.fit(train)
    ```
    Note how we pass batch size directly to the valuation method, which does
    not take a sampler since it uses MSR sampling internally.


## General semi-values

As explained above, both Beta Shapley and Banzhaf indices are special cases of
semi-values. In pyDVL we provide a general method for computing these with any
combination of the three ingredients that define a semi-value:

- A utility function $u$.
- A sampling method.
- A weighting scheme $w$.

You can construct any combination of these three ingredients with subclasses of
[SemivalueValuation][pydvl.valuation.methods.semivalue.SemivalueValuation] and
any of the samplers defined in [pydvl.valuation.samplers][].

Allowing any combination enables testing different importance-sampling schemes
and can help when experimenting with models that are more sensitive to changes
in training set size.[^bzf-stability]

For more on this topic and how Monte Carlo sampling interacts with the
semi-value coefficient and the sampler probabilities, see [[semi-values-sampling]].


[^bzf-stability]: Note however that Data Banzhaf has shown to be among the most
    robust to variance in the utility function, in the sense of rank stability,
    across a range of models and datasets [@wang_data_2023].
