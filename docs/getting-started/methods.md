---
title: Methods
alias: 
  name: methods
  text: Methods
---

We currently implement the following methods:

## Data valuation  { #implemented-methods-data-valuation }

- **LOO** implemented via
  [LOOValuation][pydvl.valuation.methods.loo.LOOValuation].

- **Permutation Shapley** (also called **ApproxShapley**), via
  [ShapleyValuation][pydvl.valuation.methods.shapley.ShapleyValuation]
  configured with a 
  [PermutationSampler][pydvl.valuation.samplers.permutation.PermutationSampler]
  [@castro_polynomial_2009].

- [**Truncated Monte Carlo Shapley**][pydvl.valuation.methods.shapley.TMCShapleyValuation],
  also available with additional truncation policies and sampling methods via
  [ShapleyValuation][pydvl.valuation.methods.shapley.ShapleyValuation]
  configured with any
  [PermutationSampler][pydvl.valuation.samplers.permutation.PermutationSampler]
  and
  [TruncationPolicy][pydvl.valuation.samplers.truncation.TruncationPolicy]
  [@ghorbani_data_2019].

- [**Data Banzhaf**][pydvl.valuation.methods.banzhaf.BanzhafValuation]
  and [**MSR sampling**][pydvl.valuation.samplers.msr.MSRSampler], in particular
  [**MSR Banzhaf**][pydvl.valuation.methods.banzhaf.MSRBanzhafValuation]
  [@wang_data_2023].

- [**Beta Shapley**][pydvl.valuation.methods.beta_shapley.BetaShapleyValuation]
  [@kwon_beta_2022].

- [**Class-Wise Shapley**][pydvl.valuation.methods.classwise_shapley.ClasswiseShapleyValuation]
  [@schoch_csshapley_2022].

- [**$\delta$-Shapley**][pydvl.valuation.methods.delta_shapley.DeltaShapleyValuation]
  [@watson_accelerated_2023]

- [**Least Core**][pydvl.valuation.methods.least_core.LeastCoreValuation]
  [@yan_if_2021].

- **Owen Sampling** with
  [ShapleyValuation][pydvl.valuation.methods.shapley.ShapleyValuation]
  configured with an [OwenSampler][pydvl.valuation.samplers.owen.OwenSampler]
  [@okhrati_multilinear_2021].

- [**Data Utility Learning**][pydvl.valuation.utility.learning.DataUtilityLearning]
  either with the
  [IndicatorUtilityModel][pydvl.valuation.utility.learning.IndicatorUtilityModel]
  from [@wang_improving_2022], or with
  [DeepSetUtilityModel][pydvl.valuation.utility.deepset.DeepSetUtilityModel]
  from [@zaheer_deep_2017].

- [**kNN-Shapley**][pydvl.valuation.methods.knn_shapley.KNNShapleyValuation]
  (exact only) [@jia_efficient_2019a].

- [**Group Testing**][pydvl.valuation.methods.gt_shapley.GroupTestingShapleyValuation]
  [@jia_efficient_2019]

- [**Data-OOB**][pydvl.valuation.methods.data_oob.DataOOBValuation]
  [@kwon_dataoob_2023].

## Influence functions  { #implemented-methods-influence-functions }

- [**CG Influence**][pydvl.influence.torch.CgInfluence].
  [@koh_understanding_2017].

- [**Direct Influence**][pydvl.influence.torch.DirectInfluence]
  [@koh_understanding_2017].

- [**LiSSA**][pydvl.influence.torch.LissaInfluence]
  [@agarwal_secondorder_2017].

- [**Arnoldi Influence**][pydvl.influence.torch.ArnoldiInfluence]
  [@schioppa_scaling_2022].

- [**EKFAC Influence**][pydvl.influence.torch.EkfacInfluence]
  [@george_fast_2018;@martens_optimizing_2015].

- [**Nyström Influence**][pydvl.influence.torch.NystroemSketchInfluence], based
  on the ideas in [@hataya_nystrom_2023] for bi-level optimization.

- [**Inverse-harmonic-mean Influence**][pydvl.influence.torch.InverseHarmonicMeanInfluence]
  [@kwon_datainf_2023].

