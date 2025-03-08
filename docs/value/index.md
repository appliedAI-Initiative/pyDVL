---
title: Data valuation
alias: 
  name: data-valuation
  text: Basics of data valuation
---

# Data valuation

!!! Info
    If you want to jump right into it, skip ahead to [Computing data
    values][computing-data-values]. If you want a quick list of applications,
    see [[data-valuation-applications]]. For a list of all algorithms
    implemented in pyDVL, see [[methods]].

**Data valuation** is the task of assigning a number to each element of a
training set which reflects its contribution to the final performance of some
model trained on it. Some methods attempt to be model-agnostic, but in most
cases the model is an integral part of the method. In these cases, this number
is not an intrinsic property of the element of interest, but typically a
function of three factors:

1. The dataset $D$, or more generally, the distribution it was sampled from: In
   some cases one only cares about values wrt. a given data set, in others
   value would ideally be the (expected) contribution of a data point to any
   random set $D$ sampled from the same distribution. pyDVL implements methods
   of the first kind.

2. The algorithm $\mathcal{A}$ mapping the data $D$ to some estimator $f$ in a
   model class $\mathcal{F}$. E.g. MSE minimization to find the parameters of a
   linear model.

3. The performance metric of interest $u$ for the problem. When value depends on
   a model, it must be measured in some way which uses it. E.g. the $R^2$ score
   or the negative MSE over a test set. This metric will be computed over a
   held-out valuation set.

pyDVL collects algorithms for the computation of data values in this sense,
mostly those derived from cooperative game theory. The methods can be found in
the package [pydvl.valuation.methods][], with support from modules like
[pydvl.valuation.samplers][] or and [pydvl.valuation.dataset][], as detailed
below.

!!! Warning
    Be sure to read the section on
    [the difficulties using data values][problems-of-data-values].

There are three main families of methods for data valuation: model-based,
influence-based and model-free. As of v0.10.0 pyDVL supports the first two.
Here, we focus on model-based (and in particular game-theoretic) concepts and
refer to the main documentation on the [influence
function][the-influence-function] for the second.

## Game theoretical methods and semi-values { #game-theoretical-methods }

The main contenders in game-theoretic approaches are [Shapley
values](shapley.md) [@ghorbani_data_2019], [@kwon_efficient_2021],
[@schoch_csshapley_2022], their generalization to so-called
[semi-values](semi-values.md) with some examples being [@kwon_beta_2022] and
[@wang_data_2023], and [the Core](the-core.md) [@yan_if_2021]. All of these are
implemented in pyDVL. For a full list see [[methods]].

In these methods, data points are considered players in a cooperative game 
whose outcome is the performance of the model when trained on subsets 
(*coalitions*) of the data, measured on a held-out **valuation set**. This 
outcome, or **utility**, must typically be computed for *every* subset of 
the training set, so that an exact computation is $\mathcal{O} (2^n)$ in the 
number of samples $n$, with each iteration requiring a full re-fitting of the 
model using a coalition as training set. Consequently, most methods involve 
Monte Carlo approximations, and sometimes approximate utilities which are 
faster to compute, e.g. proxy models [@wang_improving_2022] or constant-cost
approximations like Neural Tangent Kernels [@wu_davinz_2022].

The reasoning behind using game theory is that, in order to be useful, an
assignment of value, dubbed **valuation function**, is usually required to
fulfil certain requirements of consistency and "fairness". For instance, in some
applications value should not depend on the order in which data are considered,
or it should be equal for samples that contribute equally to any subset of the
data (of equal size). When considering aggregated value for (sub-)sets of data
there are additional desiderata, like having a value function that does not
increase with repeated samples. Game-theoretic methods are all rooted in axioms
that by construction ensure different desiderata, but despite their practical
usefulness, none of them are either necessary or sufficient for all
applications. For instance, SV methods try to equitably distribute all value
among all samples, failing to identify repeated ones as unnecessary, with e.g. a
zero value.

## Computing data values { #computing-data-values }

Using pyDVL to compute data values is a flexible process that can be broken down
into several steps. This degree of flexibility allows for a wide range of
applications, but it can also be a bit overwhelming. The following steps are
necessary:

1. Creating two [Datasets][pydvl.valuation.Dataset] object from your data: one
   for training and one for evaluating the utility. The quality of this latter set
   is crucial for the quality of the values.
2. Choosing a scoring function,typically something like accuracy or $R^2$, but
   it can be any function that takes a model and a dataset and returns a number.
   The test dataset is attached to the scorer. This is done by instantiating a
   [SupervisedScorer][pydvl.valuation.scorers.SupervisedScorer] object. Other
   types, and subclassing is possible for non-supervised problems.
3. Creating a utility object that ties your model to the scoring function. This
   is done by instantiating a [ModelUtility][pydvl.valuation.utility.ModelUtility].
4. Computing values with a valuation method of your choice, e.g. via
   [BanzhafValuation][pydvl.valuation.methods.banzhaf.BanzhafValuation]
   or [ShapleyValuation][pydvl.valuation.methods.ShapleyValuation]. For
   [semi-value][semi-values] methods, you will also need to choose a subset
   sampling scheme, e.g. a
   [PermutationSampler][pydvl.valuation.samplers.permutation.PermutationSampler]
   or a simple [UniformSampler][pydvl.valuation.samplers.powerset.UniformSampler].
5. For methods that require it, in particular those using infinite subset
   sampling schemes, one must choose a stopping criterion, that is a
   [stopping condition][pydvl.valuation.stopping] that interrupts the
   computation e.g. when the change in estimates is low, or the number of
   iterations or time elapsed exceed some threshold.


### Creating a Dataset

The first item in the tuple $(D, \mathcal{A}, u)$ characterising data value is
the dataset. The class [Dataset][pydvl.valuation.Dataset] is a simple
convenience wrapper for use across the library. Some class methods allow for the
convenient creation of train / test splits, e.g. with
[from_arrays][pydvl.valuation.Dataset.from_arrays]:

??? Example "Constructing a synthetic classification dataset"

    ```python
    from pydvl.valuation.dataset import Dataset
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=100, n_features=20, n_classes=3)
    train, test = Dataset.from_arrays(X, y, stratify_by_target=True)
    ```

With the class method
[from_sklearn][pydvl.valuation.Dataset.from_sklearn] it is possible to
construct a dataset from any of the toy datasets in [sklearn.datasets][]. 

??? Example "Loading a scikit-learn dataset"

    ```python
    from pydvl.valuation.dataset import Dataset
    from sklearn.datasets import load_iris
    train, test = Dataset.from_sklearn(
        load_iris(), train_size=0.8, stratify_by_target=True
    )
    ```

#### Grouping data

Be it because data valuation methods are computationally very expensive, or
because we are interested in the groups themselves, it can be often useful or
necessary to group samples to valuate them together.
[GroupedDataset][pydvl.valuation.dataset.GroupedDataset] provides an alternative
to [Dataset][pydvl.valuation.Dataset] with the same interface which allows this.

You can see an example in action in the
[Spotify notebook](../examples/shapley_basic_spotify), but here's a simple
example grouping a pre-existing `Dataset`. First we construct an array mapping
each index in the dataset to a group, then use
[from_dataset][pydvl.valuation.dataset.GroupedDataset.from_dataset]:

??? Example "Grouping a dataset"
    This is of course silly, but it serves to illustrate the functionality:

    ```python
    import numpy as np
    from pydvl.valuation.dataset import Dataset, GroupedDataset
    
    dataset = Dataset.from_sklearn(sk.datasets.fetch_covtype())
    n_groups = 5800
    # Randomly assign elements to any one of n_groups:
    data_groups = np.random.randint(0, n_groups, len(dataset))
    dummy_group_names = [f"Group {i}" for i in range(n_groups)]
    grouped_dataset = GroupedDataset.from_dataset(
      dataset, data_groups, dummy_group_names
    )
    ```

### Creating a utility

In pyDVL we have slightly overloaded the name "utility" and use it to refer to
an object that keeps track of both the method and its evaluation. For
model-based methods like all semi-values including Shapley, the utility
will be an instance of [ModelUtility][pydvl.valuation.utility.ModelUtility] which,
as mentioned, is a convenient wrapper for the model and scoring function.

??? Example "Creating a `ModelUtility`"

    ```python
    import sklearn as sk
    from pydvl.valuation import Dataset, ModelUtility, SupervisedScorer
    
    train, test = Dataset.from_sklearn(sk.datasets.load_iris(), train_size=0.6)
    model = sk.svm.SVC()
    # Uses the model.score() method by default
    scorer = SupervisedScorer(model, test)
    utility = ModelUtility(model, scorer)
    ```

Note how we pass the test set to the scorer. Importantly, the object provides
information about the range of the score, which is used by some methods to
estimate the number of samples necessary, and about what default value to use
when the model fails to train.

If we pass a model to [SupervisedScorer][pydvl.valuation.scorers.SupervisedScorer],
it will use the model's `score()` method by default, but it is possible to use
any scoring function (greater values must be better). In particular, the
constructor accepts the same types of arguments as those of
[sklearn.model_selection.cross_validate][]: a string, a scorer callable or
[None][] for the default.

```python
scorer = SupervisedScorer("explained_variance", default=0.0, range=(-np.inf, 1))
```

The object `utility` is a callable and is used by data valuation methods to
train the model on various subsets of the data and evaluate its performance. 
`ModelUtility` wraps the `fit()` method of the model to cache its results. In
some (rare) cases this reduces computation times of Monte Carlo methods. Because
of how caching is implemented, it is important not to reuse `ModelUtility`
objects for different datasets. You can read more about [setting up the
cache][getting-started-cache] in the installation guide, and in the
documentation of the [caching][pydvl.utils.caching] module.


### Computing some values

By far the most popular concept of value is the Shapley value, a particular case
of [semi-value][semi-values]. In order to compute them for a training set, all
we need to do after the previous steps is to instantiate a
[ShapleyValuation][pydvl.valuation.methods.ShapleyValuation] object and call its
`fit()` method.

??? Example "Computing Shapley values"
    ```python
    import sklearn as sk
    from joblib import parallel_config
    from pydvl.valuation import Dataset, ModelUtility, ShapleyValuation, SupervisedScorer
    
    train, test = Dataset.from_sklearn(sk.datasets.load_iris(), train_size=0.6)
    model = sk.svm.SVC()
    scorer = SupervisedScorer("accuracy", test, default=0.0, range=(0, 1))
    utility = ModelUtility(model, scorer)
    sampler = PermutationSampler()  # Just one of many examples
    stopping = MaxUpdates(100)  # A trivial criterion
    shapley = ShapleyValuation(utility, sampler, stopping)
    with parallel_config(n_jobs=-1):
        shapley.fit(train)

    results = shapley.values()
    ```

Note our use of [joblib.parallel_config][] in the example in order to
parallelize the computation of the values. Most valuation methods support this.

The result type of all valuations is an object of type
[ValuationResult][pydvl.valuation.result.ValuationResult]. This can be iterated
over, sliced, sorted, as well as converted to a [pandas.DataFrame][] using
[to_dataframe][pydvl.valuation.result.ValuationResult.to_dataframe].

### Learning the utility

Since each evaluation of the utility entails a full retrain of the model on a new subset of the training data, it is natural to try to learn this mapping from subsets to scores. This is the idea behind **Data Utility Learning (DUL)**
[@wang_improving_2022] and in pyDVL it's as simple as wrapping the
`ModelUtility` inside [DataUtilityLearning][pydvl.valuation.utility.DataUtilityLearning]:

```python
from pydvl.valuation import ModelUtility, DataUtilityLearning, Dataset
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.datasets import load_iris

dataset = Dataset.from_sklearn(load_iris())
u = ModelUtility(LogisticRegression(), dataset)
training_budget = 3
wrapped_u = DataUtilityLearning(u, training_budget, LinearRegression())

# First 3 calls will be computed normally
for i in range(training_budget):
   _ = wrapped_u((i,))
# Subsequent calls will be computed using the learned model for DUL
wrapped_u((1, 2, 3))
```


## Problems of data values

There are a number of factors that affect how useful values can be for your
project. In particular, regression can be especially tricky, but the particular
nature of every (non-trivial) ML problem can have an effect:

* **Variance of the utility**: Classical applications of game theoretic value
  concepts operate with deterministic utilities, as do many of the bounds in the
  literature. But in ML we use an evaluation of the model on a validation set as a
  proxy for the true risk. Even if the utility is bounded, its variance will
  affect final values, and even more so any Monte Carlo estimates.
  Several works have tried to cope with variance. [@wang_data_2023] prove that by
  relaxing one of the Shapley axioms and considering the general class of
  semi-values, of which Shapley is an instance, one can prove that a choice of
  constant weights is the best one can do in a utility-agnostic setting. This
  method, dubbed *Data Banzhaf*, is available in pyDVL as
  [BanzhafValuation][pydvl.valuation.methods.BanzhafValuation].

    ??? tip "Averaging repeated utility evaluations"
        One workaround in pyDVL is to configure the caching system to allow multiple
        evaluations of the utility for every index set. A moving average is 
        computed and returned once the standard error is small, see
        [CachedFuncConfig][pydvl.utils.caching.config.CachedFuncConfig]. Note
        however that in practice, the likelihood of cache hits is low, so one
        would have to force recomputation manually somehow.

* **Unbounded utility**: Choosing a scorer for a classifier is simple: accuracy
  or some F-score provides a bounded number with a clear interpretation. However,
  in regression problems most scores, like $R^2$, are not bounded because
  regressors can be arbitrarily bad. This leads to great variability in the
  utility for low sample sizes, and hence unreliable Monte Carlo approximations
  to the values. Nevertheless, in practice it is only the ranking of samples
  that matters, and this tends to be accurate (wrt. to the true ranking) despite
  inaccurate values.

    ??? tip "Squashing scores" 
        pyDVL offers a dedicated [function
        composition][pydvl.valuation.scorers.utils.compose_score] for scorer functions which
        can be used to squash a score. The following is defined in the module
        [scorers][pydvl.valuation.scorers]:
        ```python
        import numpy as np
        from pydvl.valuation.score import compose_score
        
        def sigmoid(x: float) -> float:
          return float(1 / (1 + np.exp(-x)))
        
        squashed_r2 = compose_score("r2", sigmoid, "squashed r2")
        
        squashed_variance = compose_score(
          "explained_variance", sigmoid, "squashed explained variance"
        )
        ```
        These squashed scores can prove useful in regression problems, but they
        can also introduce issues in the low-value regime.

* **Data set size**: Computing exact Shapley values is NP-hard, and Monte Carlo
  approximations can converge slowly. Massive datasets are thus impractical, at
  least with game-theoretical methods. A workaround is to group samples and investigate their value together. You can do this using
  [GroupedDataset][pydvl.valuation.GroupedDataset]. There is a fully
  worked-out [example here](../examples/shapley_basic_spotify). Some algorithms
  also provide different sampling strategies to reduce the variance, but due to a
  no-free-lunch-type theorem, no single strategy can be optimal for all utilities.
  Finally, model specific methods like
  [kNN-Shapley][pydvl.valuation.methods.knn_shapley] [@jia_efficient_2019a], or
  altogether different and typically faster approaches like
  [Data-OOB][pydvl.valuation.methods.data_oob.DataOOBValuation] [@kwon_dataoob_2023] can also be
  used. 

* **Model size**: Since every evaluation of the utility entails retraining the
  whole model on a subset of the data, large models require great amounts of
  computation. But also, they will effortlessly interpolate small to medium
  datasets, leading to great variance in the evaluation of performance on the
  dedicated validation set. One mitigation for this problem is cross-validation,
  but this would incur massive computational cost. As of v0.8.1 there are no
  facilities in pyDVL for cross-validating the utility (note that this would
  require cross-validating the whole value computation).


## A comment about sampling

One might fear that there is a problem because the computation of the value
$\psi_i$ requires at least some bootstrap samples *not* to include the $i$-th
sample. But we can see that this is rarely an issue, and its probability of
happening can be easily computed: For a training set of size $n$ and
bootstrapping sample size $m \le n$, the probability that index $i$ is not
included in a bootstrap sample is

$$\prod_{j=1}^m \mathbb{P}(i \text{ is not drawn at pos. } j) = (1 - 1/n)^m,$$

i.e. for each of the $m$ draws, the sample is not picked (for $m=n$ this
converges to $1/e \approx 0.368$). The probability that across $B$ bootstrapped
samples a point is not included is therefore $(1  - 1/n)^{mB}$, which is
typically extremely low.

Incidentally, this allows us to estimate the estimated number of unique indices
in a bootstrap sample of size $m$ as $m(1 - 1/n)^m$.
