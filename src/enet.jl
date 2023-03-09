# This file is a part of KCenters.jl

using SimilaritySearch
export enet

"""
    enet(dist::SemiMetric, X::AbstractDatabase, stop)

Selects a number of farthest points in `X`, using a farthest first traversal

"""
function enet(dist::SemiMetric, X::AbstractDatabase, stop::Function; verbose=true)
    N = length(X)
    irefs = Int32[]
    dmaxlist = Float32[]
    nndist = Vector{Float32}(undef, N)
    fill!(nndist, typemax(Float32))
    imax::Int = rand(1:N)
    dmax::Float32 = typemax(Float32)
    N == 0 && return (; irefs, seq=nndist, dmax)
    
    @inbounds for i in 1:N
        push!(dmaxlist, dmax)
        push!(irefs, imax)
        verbose && println(stderr, "computing fartest point $(length(irefs)), dmax: $dmax, imax: $imax, n: $(length(X))")

        pivot = X[imax]
        @batch minbatch=getminbatch(0, N) for i in 1:N
            d = evaluate(dist, X[i], pivot)
            nndist[i] = min(nndist[i], d)
        end

        dmax, imax = findmax(nndist)
        stop(dmaxlist, X) && break
    end

    (; irefs, seq=nndist, dmax)
end

"""
    enet(dist::SemiMetric, X::AbstractDatabase, k::Integer) = enet(dist, X, size_criterion(k))

Selects `k` items far from each other based on Farthest First Traversal algorithm.

Returns a named tuple \$(nn, irefs, dmax)\$.

- `irefs` contains the list of centers (indexes to ``X``)
- `seq` contains the nearest references for each object in ``X`` (in ``X`` order)
- `dmax` smallest distance among centers
"""
enet(dist::SemiMetric, X::AbstractDatabase, k::Integer) = enet(dist, X, size_criterion(k))
