---
title: Leave-One-Out values
alias: 
  name: loo-valuation
---

LOO is the simplest approach to valuation. It assigns to each sample its
*marginal utility* as value:

$$v_u(i) = u(D) - u(D_{-i}).$$

For notational simplicity, we consider the valuation function as defined over
the indices of the dataset $D$, and $i \in D$ is the index of the sample,
$D_{-i}$ is the training set without the sample $x_i$, and $u$ is the utility
function.

For the purposes of data valuation, this is rarely useful beyond serving as a
baseline for benchmarking. Although in some benchmarks it can perform
astonishingly well on occasion. One particular weakness is that it does not
necessarily correlate with an intrinsic value of a sample: since it is a
marginal utility, it is affected by diminishing returns. Often, the training set
is large enough for a single sample not to have any significant effect on
training performance, despite any qualities it may possess. Whether this is
indicative of low value or not depends on one's goals and definitions, but
other methods are typically preferable.

??? Example "Leave-One-Out values"
    ```python
    from joblib import parallel_config
    from pydvl.valuation import Dataset, LOOValuation, ModelUtility
    
    train, test = Dataset.from_arrays(...)
    model = ...
    utility = ModelUtility(model)
    val_loo = LOOValuation(utility, progress=True)
    with parallel_config(n_jobs=12):
        val_loo.fit(train)
    result = val_loo.values(sort=True)
    ```
