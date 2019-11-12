import StatsBase: fit, predict
using StatsBase
using SimilaritySearch
import SimilaritySearch: search, optimize!
using JSON
export DeloneInvIndex, fit, predict

mutable struct DeloneInvIndex{T} <: Index
    db::Vector{T}
    centers::Index
    lists::Vector{Vector{Int}}
    dmax::Vector{Float64}
    n::Int
    region_expansion::Int
end

"""
    fit(::Type{DeloneInvIndex}, X::AbstractVector{T}, kcenters_::NamedTuple, region_expansion=3) where T

Creates a DeloneInvIndex, which is a metric index using the `kcenters` output and `X`.
This is an index that implements approximate search.

"""
function fit(::Type{DeloneInvIndex}, X::AbstractVector{T}, kcenters_::NamedTuple, region_expansion=3) where T
    k = length(kcenters_.centroids)
    dmax = zeros(Float64, k)
    lists = [Int[] for i in 1:k]

    for i in eachindex(kcenters_.codes)
        code = kcenters_.codes[i]
        d = kcenters_.distances[i]
        push!(lists[code], i)
        dmax[code] = max(dmax[code], d)
    end

    C = fit(Sequential, kcenters_.centroids)
    DeloneInvIndex(X, C, lists, dmax, length(kcenters_.codes), region_expansion)
end

"""
    search(index::DeloneInvIndex{T}, dist::Function, q::T, res::KnnResult) where T

Searches nearest neighbors of `q` inside the `index` under the distance function `dist`.
"""
function search(index::DeloneInvIndex{T}, dist::Function, q::T, res::KnnResult) where T
    cres = search(index.centers, dist, q, KnnResult(index.region_expansion))
    for c in cres
        @inbounds for i in index.lists[c.objID]
            d = dist(q, index.db[i])
            push!(res, i, d)
        end
    end

    res
end


"""
    optimize!(index::DeloneInvIndex{T}, dist::Function, recall=0.9; k=10, num_queries=128, perf=nothing, verbose=false) where T

Tries to configure `index` to achieve the specified recall for fetching `k` nearest neighbors. Notice that if `perf` is not given then
the index will use dataset's items and therefore it will adjust for them.
"""
function optimize!(index::DeloneInvIndex{T}, dist::Function, recall=0.9; k=10, num_queries=128, perf=nothing, verbose=false) where T
    verbose && println("KCenters.DeloneInvIndex> optimizing for recall=$(recall)")
    if perf === nothing
        perf = Performance(index.db, dist; expected_k=k, num_queries=num_queries)
    end

    index.region_expansion = 1
    p = probe(perf, index, dist)

    while p.recall < recall && index.region_expansion < length(index.lists)
        index.region_expansion += 1
        verbose && println("KCenters.DeloneInvIndex> optimize! step region_expansion=$(index.region_expansion), performance $(JSON.json(p))")
        p = probe(perf, index, dist)
    end

    verbose && println("KCenters.DeloneInvIndex> reached performance $(JSON.json(p))")
    index
end
