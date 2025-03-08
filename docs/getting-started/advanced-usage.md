---
title: Advanced usage
alias: 
  name: advanced-usage
  text: Advanced usage
---

# Advanced usage

Besides the dos and don'ts of data valuation itself, which are the subject of
the examples and the documentation of each method, there are two main things to
keep in mind when using pyDVL namely Parallelization and Caching.

## Parallelization { #setting-up-parallelization }

pyDVL uses parallelization to scale and speed up computations. It does so
using Dask or Joblib (with any of its backends). The first is used in
the [influence][pydvl.influence] package whereas the latter is used in 
the [valuation][pydvl.valuation] package.

### Data valuation  { #data-valuation-parallelization }

For data valuation, pyDVL uses [joblib](https://joblib.readthedocs.io/en/latest/)
for transparent parallelization of most methods using any of the backends
available to joblib.

If you want to use ray or dask as backends, please follow the instructions
in [joblib's documentation][joblib.parallel_backend]. Mostly it's just a matter
of registering the backend with [joblib.register_parallel_backend][] and then
using it within the context manager [joblib.parallel_config][] around the code
that you want to parallelize, which is usually the call to the `fit` method of
the valuation object.

??? Example "Basic fitting in parallel"
    ```python
    import sklearn as sk
    from joblib import parallel_config, register_parallel_backend
    from pydvl.valuation import *
    from ray.util.joblib import register_ray

    register_ray()
    
    train, test = Dataset.from_arrays(...)
    model = sk.svm.SVC()
    scorer = SupervisedScorer("accuracy", test, default=0.0, range=(0, 1))
    utility = ModelUtility(model, scorer)
    sampler = PermutationSampler(truncation=NoTruncation())
    stopping = MinUpdates(7000) | MaxTime(3600)
    shapley = ShapleyValuation(utility, sampler, stopping, progress=True)

    with parallel_config(backend="ray", n_jobs=128):
        shapley.fit(train)

    results = shapley.values()
    ```

Note that you will have to install additional dependencies (see
[Extras][installation-extras]) and to provide a running cluster (or run ray in
local mode). For instance, for ray follow the instructions in Ray's
documentation to [set up a remote
cluster](https://docs.ray.io/en/latest/cluster/key-concepts.html). You could
alternatively use a local cluster and in that case you don't have to set
anything up.

!!! info
    As of v0.10.0 pyDVL does not allow requesting resources per task sent to the
    cluster, so you will need to make sure that each worker has enough resources
    to handle the tasks it receives. A data valuation task using game-theoretic
    methods will typically make a copy of the whole model and dataset to each
    worker, even if the re-training only happens on a subset of the data. Some
    backends, like "loky" will use memory mapping to avoid copying the dataset
    to each worker, but in general you should make sure that each worker has
    enough memory to handle the whole dataset.


### Influence functions { #influence-parallelization }

Refer to [Scaling influence computation][scaling-influence-computation] for
explanations about parallelization for Influence Functions.

## Caching { #getting-started-cache }

PyDVL can cache (memoize) the computation of the utility function and speed up
some computations for data valuation. It is however disabled by default because
single runs of methods rarely benefit much from it. When it is enabled it takes
into account the data indices passed as argument and the utility function
wrapped into the [Utility][pydvl.valuation.utility.ModelUtility] object. This
means that care must be taken when reusing the same utility function with
different data, see the documentation for the [caching
package][pydvl.utils.caching] for more information.

In general, caching won't play a major role in the computation of Shapley values
because the probability of sampling the same subset twice, and hence needing
the same utility function computation, is very low. However, **it can be very
useful when comparing methods that use the same utility function, or when
running multiple experiments with the same data**.

pyDVL supports 3 different caching backends:

- [InMemoryCacheBackend][pydvl.utils.caching.memory.InMemoryCacheBackend]:
  an in-memory cache backend that uses a dictionary to store and retrieve
  cached values. This is used to share cached values between threads
  in a single process. This backend is provided for completeness, since
  parallelization is almost never done using threads, 
- [DiskCacheBackend][pydvl.utils.caching.disk.DiskCacheBackend]:
  a disk-based cache backend that uses pickled values written to and read from
  disk. This is used to share cached values between processes in a single machine.
  !!! warning "Disk cache"
      The disk cache is a stub implementation which pickles each utility
      evaluation and is extremely inefficient. If it proves useful, we might
      implement a more efficient version in the future.
- [MemcachedCacheBackend][pydvl.utils.caching.memcached.MemcachedCacheBackend]:
  a [Memcached](https://memcached.org/)-based cache backend that uses pickled values written to
  and read from a Memcached server. This is used to share cached values
  between processes across one or multiple machines.

??? info "Memcached extras"
    The Memcached backend requires optional dependencies. See 
    [Extras][installation-extras] for more information.

Using the caches is as simple as passing the backend to the utility constructor.
Please refer to the documentation and examples of each backend class for more
details.

!!! tip "Using the cache"
    Continue reading about the cache in the documentation for the [caching
    package][pydvl.utils.caching].

### Setting up the Memcached cache { #setting-up-memcached }

[Memcached](https://memcached.org/) is an in-memory key-value store accessible
over the network.

You can either install it as a package or run it inside a docker container (the
simplest). For installation instructions, refer to the [Getting
started](https://github.com/memcached/memcached/wiki#getting-started) section in
memcached's wiki. Then you can run it with:

```shell
memcached -u user
```

To run memcached inside a container in daemon mode instead, use:

```shell
docker container run -d --rm -p 11211:11211 memcached:latest
```
