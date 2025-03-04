---
title: Getting started
alias:
  name: getting-started
  title: Getting started
---

# Getting started

If you want to jump straight in, [install pyDVL](#installation)
and then check out [[examples|the examples]]. You will probably want to install
with support for [influence function computation](#installation-influences).

We have introductions to the ideas behind [[data-valuation|Data valuation]] and
[[influence-function|Influence functions]], as well as a short overview of
[common applications](applications.md).


## Installing pyDVL { #installation }

To install the latest release use:

```shell
pip install pyDVL
```

See [Extras][installation-extras] for optional dependencies, in particular if
you are interested in influence functions. You can also install the latest
development version from [TestPyPI](https://test.pypi.org/project/pyDVL/):

```shell
pip install pyDVL --index-url https://test.pypi.org/simple/
```

In order to check the installation you can use:

```shell
python -c "import pydvl; print(pydvl.__version__)"
```

## Dependencies

pyDVL requires Python >= 3.9, [numpy](https://numpy.org/),
[scikit-learn](https://scikit-learn.org/stable/), [scipy](https://scipy.org/),
[cvxpy](https://www.cvxpy.org/) for the core methods, and
[joblib](https://joblib.readthedocs.io/en/stable/) for parallelization locally.
Additionally,the [Influence functions][pydvl.influence] module requires PyTorch
(see [Extras][installation-extras] below).


## Extras { #installation-extras }

pyDVL has a few [extra](https://peps.python.org/pep-0508/#extras) dependencies
that can be optionally installed:

### Influence functions { #installation-influences }

To use the module on influence functions, [pydvl.influence][pydvl.influence],
run:

```shell
pip install pyDVL[influence]
```

This includes a dependency on [PyTorch](https://pytorch.org/) (Version 2.0 and
above) and thus is left out by default.

### CuPy

In case that you have a supported version of CUDA installed (v11.2 to 11.8 as of
this writing), you can enable eigenvalue computations for low-rank approximations
with [CuPy](https://docs.cupy.dev/en/stable/index.html) on the GPU by using:

```shell
pip install pyDVL[cupy]
```

This installs [cupy-cuda11x](https://pypi.org/project/cupy-cuda11x/).

If you use a different version of CUDA, please install CuPy
[manually](https://docs.cupy.dev/en/stable/install.html).

### Ray

If you want to use [Ray](https://www.ray.io/) to distribute data valuation
workloads across nodes in a cluster (it can be used locally as well, but for
this we recommend joblib instead) install pyDVL using:

```shell
pip install pyDVL[ray]
```

see [the intro to parallelization][setting-up-parallelization] for more
details on how to use it.

### Memcached

If you want to use [Memcached](https://memcached.org/) for caching utility
evaluations, use:

```shell
pip install pyDVL[memcached]
```

This installs [pymemcache](https://github.com/pinterest/pymemcache)
additionally. Be aware that you still have to start a memcached server manually.
See [Setting up the Memcached cache][setting-up-memcached].
