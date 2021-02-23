```@meta
CurrentModule = KCenters
```


# KCenters.jl

The `KCenters` package implements algorithms that approximate the kcenters problem, in particular, we use farthest first traversal and density nets. It adds several variants of center selection and stopping criterions, along with a number of heuristics and utilities for taking advantage of the resulting groups.

- The majority of tasks will use [`kcenters`](@ref) that outputs a [`ClusteringData`](@ref).
- Low level tasks may require to use [`enet`](@ref) or [`dnet`](@ref).
- Partitioning data differently to [`ClusteringData`](@ref).
- For more detailed usage, please visit each one of the pages on the left panel.