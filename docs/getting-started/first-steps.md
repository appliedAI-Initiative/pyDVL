---
title: First steps
alias: 
  name: first-steps
  text: First Steps
---

# First steps

!!! info
    Make sure you have read [[getting-started#installation]] before using the
    library. In particular read about which extra dependencies you may need.

## Main concepts

pyDVL aims to be a repository of production-ready, reference implementations of
algorithms for data valuation and influence functions. Even though we only
briefly introduce key concepts in the documentation, the following sections 
should be enough to get you started.

<div class="grid cards" markdown>

-   [[data-valuation]]
  
    ---
    Key objects and usage patterns for Shapley values and related methods.
  
    [[data-valuation|:octicons-arrow-right-24: Data valuation]]
  
-   [[influence-function]]
  
    ---
    Instructions on how to compute influence functions, and many approximations.
    
    [[influence-function|:octicons-arrow-right-24: Influence functions]]

</div>

## Running the examples

If you are somewhat familiar with the concepts of data valuation, you can start
by browsing our worked-out examples illustrating pyDVL's capabilities either:

- In the examples under [[data-valuation]] and [[influence-function]].
- Using [binder](https://mybinder.org/) notebooks, deployed from each
  example's page.
- Locally, by starting a jupyter server at the root of the project. You will
  have to install jupyter first manually since it's not a dependency of the
  library.

## Advanced usage

Refer to the [[advanced-usage]] page for explanations on how to enable
and use parallelization and caching.
