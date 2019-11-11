import StatsBase: fit, predict
using StatsBase
using SimilaritySearch
import SimilaritySearch: search, optimize!
export InvIndex, fit, predict

mutable struct InvIndex{T} <: Index
    db::Vector{T}
    centers::Index
    lists::Vector{Vector{Int}}
    dmax::Vector{Float64}
    n::Int
    region_expansion::Int
end

function fit(::Type{InvIndex}, X::AbstractVector{T}, kcenters_::NamedTuple, region_expansion=3) where T
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
    InvIndex(X, C, lists, dmax, length(kcenters_.codes), region_expansion)
end

function search(index::InvIndex{T}, dist::Function, q::T, res::KnnResult) where T
    cres = search(index.centers, dist, q, KnnResult(index.region_expansion))
    for c in cres
        @inbounds for i in index.lists[c.objID]
            d = dist(q, index.db[i])
            push!(res, i, d)
        end
    end

    res
end

function optimize!(index::InvIndex{T}, dist::Function, recall=0.9; k=10, num_queries=128, perf=nothing, verbose=false) where T
    verbose && println("KCenters.InvIndex> optimizing for recall=$(recall)")
    if perf === nothing
        perf = Performance(index.db, dist; expected_k=k, num_queries=num_queries)
    end

    index.region_expansion = 1
    p = probe(perf, index, dist)

    while p.recall < recall && index.region_expansion < length(index.lists)
        index.region_expansion += 1
        verbose && println("KCenters.InvIndex> optimize! step region_expansion=$(index.region_expansion), performance $(p)")
        p = probe(perf, index, dist)
    end

    verbose && println("KCenters.InvIndex> reached performance $(p)")
    
    return index
end
