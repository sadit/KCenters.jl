```@meta
CurrentModule = KCenters
DocTestSetup = quote
    using KCenters
end
```

# Center selection schemes

KCenters support different ways to compute or select object prototypes (centers).
You can use them through [`center`](@ref) function; any new scheme must specialize
from [`AbstractCenterSelection`](@ref) and specialize `center` function.

```@docs
AbstractCenterSelection
CentroidSelection
RandomCenterSelection
MedoidSelection
KnnCentroidSelection
center
```