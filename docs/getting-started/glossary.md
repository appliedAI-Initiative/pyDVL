# Glossary

This glossary is meant to provide only brief explanations of each term, helping
to clarify the concepts and techniques used in the library. For more detailed
information, please refer to the relevant literature or resources.

!!! warning
    This glossary is still a work in progress. Pull requests are welcome!

Terms in data valuation and influence functions:

### Arnoldi Method { #glossary-arnoldi }

The Arnoldi method approximately computes eigenvalue, eigenvector pairs of
a symmetric matrix. For influence functions, it is used to approximate
the [iHVP][glossary-iHVP].
Introduced by [@schioppa_scaling_2022] in the context of influence functions.

  * [Implementation (torch)
    ][pydvl.influence.torch.influence_function_model.ArnoldiInfluence]
  * [Documentation (torch)][arnoldi]

### Block Conjugate Gradient { #glossary-block-cg }

A blocked version of [CG][glossary-conjugate-gradient], which solves several linear
systems simultaneously. For Influence Functions, it is used to
approximate the [iHVP][glossary-iHVP].

 * [Implementation (torch)
   ][pydvl.influence.torch.influence_function_model.CgInfluence]
 * [Documentation (torch)][cg]

### Class-wise Shapley { #glossary-class-wise-shapley }

Class-wise Shapley is a Shapley valuation method which introduces a utility
function that balances in-class, and out-of-class accuracy, with the goal of
favoring points that improve the model's performance on the class they belong
to. It is estimated to be particularly useful in imbalanced datasets, but more
research is needed to confirm this.
Introduced by [@schoch_csshapley_2022].

 * [Implementation
   ][pydvl.valuation.methods.classwise_shapley.ClasswiseShapleyValuation]
 * [Documentation][class-wise-shapley]

### Conjugate Gradient { #glossary-cg }

CG is an algorithm for solving linear systems with a symmetric and
positive-definite coefficient matrix. For Influence Functions, it is used to
approximate the [iHVP][glossary-iHVP].

 * [Implementation
   (torch)][pydvl.influence.torch.influence_function_model.CgInfluence]
 * [Documentation (torch)][cg]

### Data-OOB { #glorssary-data-oob }

Data-OOB is a method for valuing data points for a bagged model using its
out-of-bag performance estimate. It overcomes the computational bottleneck of
Shapley-based methods by evaluating each weak learner in an ensemble over
samples it hasn't seen during training, and averaging the performance across
all weak learners.
Introduced in [@kwon_dataoob_2023].

 * [Implementation][pydvl.valuation.methods.data_oob.DataOOBValuation]
 * [Documentation][data-oob]


### Data Utility Learning { #glossary-data-utility-learning }

Data Utility Learning is a method that uses an ML model to learn the utility
function. Essentially, it learns to predict the performance of a model when
trained on a given set of indices from the dataset. The cost of training this
model is quickly amortized by avoiding costly re-evaluations of the original
utility.
Introduced by [@wang_improving_2022].

 * [Implementation][pydvl.valuation.utility.learning.DataUtilityLearning]
 * [Documentation][creating-a-utility]

### Eigenvalue-corrected Kronecker-Factored Approximate Curvature

EKFAC builds on [K-FAC][flossary-k-fac] by correcting for the approximation
errors in the eigenvalues of the blocks of the Kronecker-factored approximate
curvature matrix. This correction aims to refine the accuracy of natural
gradient approximations, thus potentially offering better training efficiency
and stability in neural networks.

 * [Implementation
   (torch)][pydvl.influence.torch.influence_function_model.EkfacInfluence]
 * [Documentation (torch)][eigenvalue-corrected-k-fac]


### Group Testing { #glossary-group-testing }

Group Testing is a strategy for identifying characteristics within groups of
items efficiently, by testing groups rather than individuals to quickly narrow
down the search for items with specific properties.
Introduced into data valuation by [@jia_efficient_2019a].

 * [Implementation][pydvl.valuation.methods.gt_shapley.GroupTestingShapleyValuation]
 * [Documentation][group-testing]

### Influence Function { #glossary-influence-function }

The Influence Function measures the impact of a single data point on a
statistical estimator. In machine learning, it's used to understand how much a
particular data point affects the model's prediction.
Introduced into data valuation by [@koh_understanding_2017].

 * [[influence-function|Documentation]]

### Inverse Hessian-vector product { #glossary-iHVP }

iHVP is the operation of calculating the product of the inverse Hessian matrix
of a function and a vector, without explicitly constructing nor inverting the
full Hessian matrix first. This is essential for influence function computation.

### Kronecker-Factored Approximate Curvature { #glossary-k-fac }

K-FAC is an optimization technique that approximates the Fisher Information
matrix's inverse efficiently. It uses the Kronecker product to factor the
matrix, significantly speeding up the computation of natural gradient updates
and potentially improving training efficiency.

### Least Core { #glossary-least-core }

The Least Core is a solution concept in cooperative game theory, referring to
the smallest set of payoffs to players that cannot be improved upon by any
coalition, ensuring stability in the allocation of value. In data valuation,
it implies solving a linear and a quadratic system whose constraints are
determined by the evaluations of the utility function on every subset of the
training data.
Introduced as data valuation method by [@yan_if_2021].

 * [Implementation][pydvl.valuation.methods.least_core.LeastCoreValuation]
 * [Documentation][least-core-values]

### Linear-time Stochastic Second-order Algorithm { #glossary-lissa }

LiSSA is an efficient algorithm for approximating the inverse Hessian-vector
product, enabling faster computations in large-scale machine learning
problems, particularly for second-order optimization.
For Influence Functions, it is used to
approximate the [iHVP][glossary-iHVP].
Introduced by [@agarwal_secondorder_2017].

 * [Implementation (torch)
   ][pydvl.influence.torch.influence_function_model.LissaInfluence]
 * [Documentation (torch)
   ][linear-time-stochastic-second-order-approximation-lissa]

### Leave-One-Out { #glossary-loo }

LOO in the context of data valuation refers to the process of evaluating the
impact of removing individual data points on the model's performance. The
value of a training point is defined as the marginal change in the model's
performance when that point is removed from the training set.

 * [Implementation][pydvl.valuation.methods.loo.LOOValuation]
 * [Documentation][leave-one-out-values]

### Maximum Sample Reuse  { #glossary-msr }

MSR is a sampling method for data valuation that updates the value of every
data point in one sample. This method can achieve much faster convergence. It
can be used with any
[semi-value][pydvl.valuation.methods.semivalue.SemivalueValuation] by setting
the sampler to be `MSR`.
Introduced by [@wang_data_2023]

* [Implementation][pydvl.valuation.samplers.msr.MSRSampler]


### Monte Carlo Least Core  { #glossary-mclc }

MCLC is a variation of the Least Core that uses a reduced amount of
constraints, sampled randomly from the powerset of the training data.
Introduced by [@yan_if_2021].

 * [Implementation][pydvl.valuation.methods.least_core.MonteCarloLeastCoreValuation]
 * [Documentation][monte-carlo-least-core]

### Monte Carlo Shapley  { #glossary-monte-carlo-shapley }

MCS estimates the Shapley Value using a Monte Carlo approximation to the sum
over subsets of the training set. This reduces computation to polynomial time
at the cost of accuracy, but this loss is typically irrelevant for downstream
applications in ML.
Introduced into data valuation by [@ghorbani_data_2019].

 * [Implementation][pydvl.valuation.methods.shapley.ShapleyValuation]
 * [Documentation][monte-carlo-combinatorial-shapley]

### Nyström Low-Rank Approximation  { #glossary-nystroem }

The Nyström approximation computes a low-rank approximation to a symmetric
positive-definite matrix via random projections. For influence functions, 
it is used to approximate the [iHVP][glossary-iHVP].
Introduced as sketch and solve algorithm in [@hataya_nystrom_2023], and as
preconditioner for [PCG][glossary-preconditioned-conjugate-gradient] in
[@frangella_randomized_2023].

 * [Implementation Sketch-and-Solve
   (torch)][pydvl.influence.torch.influence_function_model.NystroemSketchInfluence]
 * [Documentation Sketch-and-Solve (torch)][nystrom-sketch-and-solve]
 * [Implementation Preconditioner
   (torch)][pydvl.influence.torch.preconditioner.NystroemPreconditioner]

### Point removal task  { #glossary-point-removal-task }

A task in data valuation where the quality of a valuation method is measured
through the impact of incrementally removing data points on the model's
performance, where the points are removed in order of their value. See

 * [Implementation][pydvl.reporting.point_removal.run_removal_experiment]
 * [Benchmarking tasks][benchmarking-tasks]

### Preconditioned Block Conjugate Gradient  { #glossary-preconditioned-block-cg }

A blocked version of [PCG][glossary-preconditioned-conjugate-gradient], which solves 
several linear systems simultaneously. For Influence Functions, it is used to
approximate the [iHVP][glossary-iHVP].

 * [Implementation CG (torch)
   ][pydvl.influence.torch.influence_function_model.CgInfluence]
 * [Implementation Preconditioner (torch)
   ][pydvl.influence.torch.pre_conditioner]
 * [Documentation (torch)][cg]

### Preconditioned Conjugate Gradient  { #glossary-preconditioned-cg }

A preconditioned version of [CG][glossary-conjugate-gradient] for improved
convergence, depending on the characteristics of the matrix and the
preconditioner. For Influence Functions, it is used to
approximate the [iHVP][glossary-iHVP].

 * [Implementation CG (torch)
   ][pydvl.influence.torch.influence_function_model.CgInfluence]
 * [Implementation Preconditioner (torch)
   ][pydvl.influence.torch.pre_conditioner]
 * [Documentation (torch)][cg]

### Shapley Value  { #glossary-shapley-value }

Shapley Value is a concept from cooperative game theory that allocates payouts
to players based on their contribution to the total payoff. In data valuation,
players are data points. The method assigns a value to each data point based
on a weighted average of its marginal contributions to the model's performance 
when trained on each subset of the training set. This requires
$\mathcal{O}(2^{n-1})$ re-trainings of the model, which is infeasible for even
trivial data set sizes, so one resorts to approximations like TMCS.
Introduced into data valuation by [@ghorbani_data_2019].

 * [Implementation][pydvl.valuation.methods.shapley]
 * [Documentation][shapley-value]

### Truncated Monte Carlo   { #glossary-tmcs }

TMCS is an efficient approach to estimating the Shapley Value using a
truncated version of the Monte Carlo method, reducing computation time while
maintaining accuracy in large datasets. Being a heuristic to permutation sampling
in Shapley valuation, it can be implemented by using a
[RelativeTruncation ][pydvl.valuation.samplers.truncation.RelativeTruncation]
for a [PermutationSampler][pydvl.valuation.samplers.permutation.PermutationSampler]
when configuring [ShapleyValuation][pydvl.valuation.methods.shapley.ShapleyValuation].
Introduced by [@ghorbani_data_2019].

 * [Implementation][pydvl.valuation.methods.shapley.ShapleyValuation]
 * [Documentation][permutation-shapley]

### Weighted Accuracy Drop  { #glossary-wad }

WAD is a metric to evaluate the impact of sequentially removing data points on
the performance of a machine learning model, weighted by their rank, i.e. by the
time at which they were removed.
Introduced by [@schoch_csshapley_2022].

---

## Other terms

### Coefficient of Variation

CV is a statistical measure of the dispersion of data points in a data series
around the mean, expressed as a percentage. It's used to compare the degree of
variation from one data series to another, even if the means are drastically
different.


### Constraint Satisfaction Problem

A CSP involves finding values for variables within specified constraints or
conditions, commonly used in scheduling, planning, and design problems where
solutions must satisfy a set of restrictions.

### Out-of-Bag

OOB refers to data samples in an ensemble learning context (like random forests)
that are not selected for training a specific model within the ensemble. These
OOB samples are used as a validation set to estimate the model's accuracy,
providing a convenient internal cross-validation mechanism.

### Machine Learning Reproducibility Challenge

The [MLRC](https://reproml.org/) is an initiative that encourages the
verification and replication of machine learning research findings, promoting
transparency and reliability in the field. Papers are published in
[Transactions on Machine Learning Research](https://jmlr.org/tmlr/) (TMLR).
