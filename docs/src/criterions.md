```@meta
CurrentModule = KCenters
DocTestSetup = quote
    using KCenters
end
```

# Stop criterions for $\epsilon$-net [`enet`](@ref) 

The following functions create an stop criterion function of the form: `(dmaxlist, database) -> bool`. This function returns `true` whenever the caller must stop and `false` otherwise.

```@index
```

```@docs
size_criterion
sqrt_criterion
change_criterion
fun_criterion
log_criterion
epsilon_criterion
salesman_criterion
```