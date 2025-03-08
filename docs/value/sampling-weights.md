---
title: Sampling strategies for semi-values
alias:
  name: semi-values-sampling
---

# Sampling strategies for semi-values { #semi-values-sampling }

!!! info
    This page explains the interaction between sampler _weights_ (probability
    of sampling sets of a given size), semi-value coefficients and Monte Carlo
    sampling.

Samplers define so-called *sampling strategies*, which subclass
[EvaluationStrategy][pydvl.valuation.samplers.EvaluationStrategy]. These
strategies are where sub-processes compute marginal utilities, by whichever
method is required depending on the sampling scheme. Crucially, they compute
**the product of a semi-value coefficient with a sampling weight**. This can be
seen either as a form of importance sampling, or as a mechanism to allow
mix-and-matching of sampling strategies and semi-value coefficients.

It is however an unnecessary step when the sampling distribution yields exactly
the semi-value coefficient, as we explain below.

??? Example "The core semi-value computation for powerset sampling"
    This is the core of the marginal update computation in
    [PowersetEvaluationStrategy][pydvl.valuation.samplers.powerset.PowersetEvaluationStrategy]:*
    ```python
    for sample in batch:
      u_i = self.utility(sample.with_idx_in_subset())
      u = self.utility(sample)
      marginal = u_i - u
      sign = np.sign(marginal)
      log_marginal = -np.inf if marginal == 0 else np.log(marginal * sign)
      log_marginal += self.log_correction(self.n_indices, len(sample.subset))
      updates.append(ValueUpdate(sample.idx, log_marginal, sign))
    ```

We will be discussing the `log_correction(n, k)`. In pyDVL we allow for almost 
arbitrary combinations of semi-value and sampler, but allow switching off the 
coefficients and provide dedicated classes that do so when possible.

## Uniform sampling

We begin by rewriting the combinatorial definition of Shapley value:

$$
\begin{eqnarray*}
  v_{\operatorname{sh}} (i) & = & \sum_{S \subseteq D_{- i}} \frac{1}{n}
  \binom{n - 1}{| S |}^{- 1}  [U (S_{+ i}) - U (S)],\\\
  & = & \sum_{S \subseteq D_{- i}} w_{\operatorname{sh}} (| S |) \delta_{i} (S),
\end{eqnarray*}
$$

where $w_{\operatorname{sh}} (| S |) = \frac{1}{n}  \binom{n - 1}{| S |}^{- 
1}$ is the Shapley weight and $\delta_{i} (S) := U (S_{+ i}) - U (S)$ the 
marginal utility. The naive Monte Carlo approximation is then to sample $S_{j} 
\sim \mathcal{U} (D_{- i})$ and let

<span id="mc-shapley" class="tm-eqlabel"></span>

$$
\begin{equation}
  \hat{v}_{\operatorname{sh}, \operatorname{unif}} (i) = \frac{1}{M}  \sum_{j
  = 1}^M w_{\operatorname{sh}} (| S_{j} |) \delta_{i} (S_{j})
  \label{mc-shapley}\tag{1}
\end{equation}
$$

However, as the number of samples $M$ grows:

$$
\begin{eqnarray*}
  \hat{v}_{\operatorname{sh}, \operatorname{unif}} (i) & \underset{M
  \rightarrow \infty}{\longrightarrow} & \underset{S \sim \mathcal{U} (D_{-
  i})}{\mathbb{E}} [w_{\operatorname{sh}} (| S |) \delta_{i} (S)]\\\
  & = & \sum_{S \subseteq D_{- i}} w_{\operatorname{sh}} (k) \delta_{i}
  (S) p_{\mathcal{U}} (S),
\end{eqnarray*}
$$

where $p_{\mathcal{U}} (S) = \frac{1}{2^{n - 1}}$ is the probability of 
sampling a set $S$ under the uniform distribution $\mathcal{U} (D_{- i}).$ 
Now, note how this is not exactly what we need, so we must account for it by 
introducing an additional coefficient in the Monte Carlo sums. We can call this 
coefficient

$$ w_{\operatorname{unif}} (S) \equiv 2^{n - 1} . $$

The product $w_{\operatorname{unif}}  (S) w_{\operatorname{sh}} (S)$ is the 
`log_correction(n, k)` in the code above. Because of how samplers work the 
coefficients only depend on the size $k = | S |$ of the subsets, and it will 
always be the inverse of the probability of a set $S,$ given that it has size 
$k.$

At every step of the MC algorithm we do the following:

!!! abstract "Monte Carlo Shapley Update"
    1. sample $S_{j} \sim \mathcal{U} (D_{- i}),$ let $k = | S_{j} |$
    1. compute the marginal $\delta_{i} (S_{j})$
    1. compute the product of coefficients for the sampler and the method: 
       $w_{\operatorname{unif}} (k) w_{\operatorname{sh}} (k)$
    1. update the running average for $\hat{v}_{\operatorname{unif}, 
       \operatorname{sh}}$

## Picking a different distribution

In our correction above, we compute

$$
w_{\operatorname{unif}} (k) w_{\operatorname{sh}} (k) = 
\frac{2^{n - 1}}{n}  \binom{n - 1}{k}^{- 1},
$$

for every set size $k.$ Even for moderately large values of $n$ and $k$ these 
are huge numbers that will introduce errors in the computation. One way of 
alleviating the problem that is employed in pyDVL is to perform all 
computations in log space and use the log-sum-exp trick for numerical 
stability.

However, while this greatly increases numerical range and accuracy, it remains 
suboptimal. What if one chose instead the sampling procedure such that $p (S) = 
w_{\operatorname{sh}} (S)$? This is the main contribution of several works in 
the area, like TMCS, AME or Owen-Shapley.

### An introductory example

A simple idea is to include indices $j$ in $S$ following an i.i.d. Bernoulli 
sampling procedure. Sample $n$ r.v.s. $X_{j} \sim \operatorname{Ber} \left( q 
= 1 / 2 \right)$ and let $S := \lbrace j : X_{j} = 1, j \neq i \rbrace.$ 
Because of the independent sampling:

$$ 
p (S) = \left( \tfrac{1}{2}  \right)^k  \left( 1 - \tfrac{1}{2} 
\right)^{m -
   k} = \tfrac{1}{2^m}.
$$

where $k = | S |$ and $m = n - 1.$ We see that for our constant case $q = 1 / 
2,$ one recovers exactly the uniform distribution $\mathcal{U} (D_{- i}),$ so 
we haven't gained much.

### A better approach

The generalization of the previous idea is dubbed AME. This is a two-stage 
sampling procedure which first samples $q \in (0, 1)$ according to some 
distribution $\mathcal{Q}$ and then samples sets $S \subseteq D_{- i}$ as 
above, with an i.i.d. process using a Bernoulli of parameter $q$: $X_{1}, 
\ldots, X_{n} \sim \operatorname{Ber} (q)$ and $S := \lbrace j : X_{j} = 1, j 
\neq i \rbrace.$ With this method we have

$$ p (S|q) = q^k  (1 - q)^{m - k}, $$

and if we pick a uniform $q \sim \mathcal{U} ((0, 1))$ and marginalize over 
$q,$ a Beta function appears, and:

$$
\begin{eqnarray*}
  p (S) & = & \int_{0}^1 q^k  (1 - q)^{m - k} \mathrm{d} q\\\
  & = & \frac{\Gamma (k + 1) \Gamma (m - k + 1)}{\Gamma (m + 2)}\\\
  & = & \frac{k! (m - k) !}{(m + 1) !}\\\
  & = & \frac{1}{n}  \binom{n - 1}{k}^{- 1}\\\
  & = & w_{\operatorname{shap}} (k) .
\end{eqnarray*}
$$

Now a Monte Carlo approximation $\hat{v}_{\operatorname{shap}, 
\operatorname{ame} (\mathcal{U})}$ like that of [(1)](#mc-shapley) converges 
exactly to $v_{\operatorname{shap}}$ **without any correcting factors**.

Formally, AME is defined as the expected marginal utility over the joint 
distribution. Let $f$ be the density of $\mathcal{Q},$ and let $\mathcal{L} (q, 
D_{- i})$ be the law of the i.i..d. Bernoulli sampling process described above 
for fixed $q,$ and $\mathcal{L}_{Q} (D_{- i})$ that of the whole procedure. 
By total expectation:

$$
\begin{eqnarray*}
  v_{\operatorname{ame}, \mathcal{Q}} (i) & := & \mathbb{E}_{S \sim
  \mathcal{L}_{\mathcal{Q}} (D_{- i})} [\delta_{i} (S)]\\\
  & = & \mathbb{E}_{q \sim \mathcal{Q}}  [\mathbb{E}_{S \sim \mathcal{L} (q,
  D_{- i})} [\delta_{i} (S) |q]]\\\
  & = & \sum_{S \subseteq D_{- i}} \delta_{i} (S)  \int_{0}^1 p (S|q) f
  (q) \mathrm{d} q
\end{eqnarray*}
$$

and

$$ 
\int_{0}^1 p (S|q) f (q) \mathrm{d} q = \int_{0}^1 q^k  (1 - q)^{m - k} 
f(q) \mathrm{d} q = p (S). 
$$

One can also pick a Beta distribution for $\mathcal{Q}$ and recover 
Beta-Shapley. For all the choices, and for the approximate AME method using 
sparse regression to estimate values from fewer utility evaluations, consult 
the documentation.

### Sampling permutations

Consider now the alternative choice of uniformly sampling permutations
$\sigma_{j} \sim \mathcal{U} (\Pi (D)).$ Recall that we let $S_{i}^{\sigma}$ be
the set of indices preceding $i$ in the permutation $\sigma.$ We want to
determine what the correction for permutation sampling $w_{\operatorname{per}}$
and Shapley values should be:[^1]

$$
\begin{eqnarray*}
  \hat{v}_{\operatorname{sh}, \operatorname{per}} (i) & = & \frac{1}{M}
  \sum_{j = 1}^M w_{\operatorname{sh}} (| S_{i}^{\sigma_{j}} |) \delta
  _{i} (S_{i}^{\sigma_{j}})\\\
  & \underset{M \rightarrow \infty}{\longrightarrow} & \underset{\sigma \sim
  \mathcal{U} (\Pi (D))}{\mathbb{E}} [w_{\operatorname{sh}} (| S_{i}^{\sigma}
  |) \delta_{i} (S_{i}^{\sigma})]\\\
  & = & \frac{1}{n!}  \sum_{\sigma \in \Pi (D)} w_{\operatorname{sh}} (|
  S_{i}^{\sigma} |)  [U (S_{i}^{\sigma} \cup \lbrace i \rbrace) - U
  (S_{i}^{\sigma})]\\\
  & \overset{(\star)}{=} & \frac{1}{n!}  \sum_{S \subseteq D_{- i}}
  w_{\operatorname{sh}} (| S |)  (n - 1 - | S |) ! | S | ! [U (S_{+ i}) - U
  (S)]\\\
  & = & \sum_{S \subseteq D_{- i}} w_{\operatorname{sh}} (| S |)
  \frac{1}{n}  \binom{n - 1}{| S |}^{- 1} \delta_{i} (S)\\\
  & = & \sum_{S \subseteq D_{- i}} w_{\operatorname{sh}} (| S |)
  w_{\operatorname{sh}} (| S |) \delta_{i} (S) .
\end{eqnarray*}
$$

We have a duplicate $w_{\operatorname{sh}}$ coefficient! It's a bit hidden in 
the algebra, but it is in fact the probability of sampling a set $S$ given that 
it has size $k.$ If we denote it as $p (S|k)$ with some abuse of notation, then:

$$ p (S|k) = \frac{1}{n}  \binom{n - 1}{| S |}^{- 1}. $$

To go back to our estimator, we could simply leave out the 
$v_{\operatorname{sh}}$ from the Monte Carlo sums and the estimator would 
naturally converge to the Shapley values. We see therefore that for permutation 
sampling we can avoid correction factors and simply compute an average of 
marginals, and we will recover $v_{\operatorname{sh}}.$

Here pyDVL allows two choices:

The simplest one is to use
[TMCShapleyValuation][pydvl.valuation.methods.shapley.TMCShapleyValuation], a
class that defines its own permutation sampler and sets its valuation
coefficient to `None`, thus signalling the Monte Carlo update procedure *not* to
apply any correction. This is preferred e.g. to reproduce the paper's results.

The alternative is to mix a generic
[ShapleyValuation][pydvl.valuation.methods.shapley.ShapleyValuation] with any
[PermutationSampler][pydvl.valuation.samplers.permutation.PermutationSampler]
configuration. Then the corrections will be applied:

$$
\hat{v}_{\operatorname{sh}, \operatorname{per}} (i) := \frac{1}{m}
\sum_{j = 1}^m w_{\operatorname{per}} (k) w_{\operatorname{sh}} (k)
\delta_{i} (S_{i}^{\sigma_{j}}), \quad k = | S_{i}^{\sigma_{j}} | 
$$

with $w_{\operatorname{per}} (k) = w_{\operatorname{sh}} (k)^{- 1},$ in order to
obtain an unbiased estimator. This also means that we will always cancel the
weights coming from the permutation sampling and can multiply with other
coefficients from different methods, for instance
[BanzhafValuation][pydvl.valuation.methods.banzhaf.BanzhafValuation].[^2]

These same choices apply to
[MSRBanzhafValuation][pydvl.valuation.methods.banzhaf.MSRBanzhafValuation] and a
few other “pre-packaged” methods.

## Conclusion: The general case and importance sampling

Let's look now at general semi-values, which are of the form:

$$
\begin{eqnarray*}
  v_{\operatorname{semi}} (i) & = & \sum_{k = 0}^{n - 1} w (k)  \sum_{S
  \subseteq D_{- i}^{(k)}} \delta_{i} (S),
\end{eqnarray*}
$$

where $D^{(k)}_{- i} := \lbrace S \subseteq D_{- i} : | S | = k \rbrace$ and 
$\sum w (k) = 1.$ Different weight choices lead to different notions of value. 
For instance, with $w (k) = w_{\operatorname{sh}}$ we have Shapley values, and 
so on.

Our discussion above tried to explain that we have a general way of mixing 
sampling strategies and semi-value coefficients. There are three quantities in 
play:

1. The semi-value coefficient $w_{\operatorname{semi}}.$
1. The sampling probability emerging in the Monte Carlo process $p (S|k).$
1. Potentially, a correction factor, $w_{\operatorname{sampler}} := p (S|k)^{- 
   1}.$

When $p (S|k) = w_{\operatorname{semi}}$ we can tell pyDVL to disable 
correction factors by subclassing from
[SemivalueValuation][pydvl.valuation.methods.semivalue.SemivalueValuation] and
overriding the property
[log_coefficient][pydvl.valuation.methods.semivalue.SemivalueValuation.log_coefficient]
to return `None`.

Alternatively, we can mix and match samplers, effectively performing importance 
sampling. Let $\mathcal{L}$ be the law of a sampling procedure such that 
$p_{\mathcal{L}} (S|k) = w_{\operatorname{semi}}$ for some semi-value 
coefficient, and let $\mathcal{Q}$ be that of any sampler we choose: Then:

$$ v_{\operatorname{semi}} (i) = \mathbb{E}_{\mathcal{L}} [\delta_{i} (S)]
   = \mathbb{E}_{Q} \left[ \frac{w_{\operatorname{semi}} (S)}{p_{Q} (S|k)}
   \delta_{i} (S) \right] $$

The drawback is that a direct implementation with that much cancelling of 
coefficients might be inefficient or numerically unstable. Integration issues 
arise to compute $p_{Q} (S|k)$ and so on. On the flip side, we can implement 
any sampling method, like antithetic sampling, and immediately benefit in all 
semi-value computations.

[^1]: At step $(\star)$ we have counted the number of permutations before a
fixed position of index $i$ and after it, because the utility does not depend on
ordering. This allows us to sum over sets instead of permutations.

[^2]: From a numerical point of view, this can be suboptimal if only used to
cancel out terms, since even in log-space, it can induce errors (which are
nevertheless lost in the general noise of computing values, but that's another
story). However, it makes the implementation very homogeneous and allows for
experimentation with different importance sampling schemes.
