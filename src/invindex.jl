import StatsBase: fit, predict
using StatsBase
using SimilaritySearch
import SimilaritySearch: search
export InvIndex, fit, predict

mutable struct InvIndex{T} <: Index
    db::Vector{T}
    centers::Index
    lists::Vector{Int}
    dmax::Vector{Float64}
    n::Int
    k::Int
end

function fit(::Type{InvIndex}, X::AbstractVector{T}, kcenters_::NamedTuple, k=3) where T
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
    InvIndex(X, C, lists, dmax, length(kcenters_.codes), k)
end

function search(index::InvIndex{T}, dist::Function, q::T, res::KnnResult) where T
    C = index.centers
    cres = search(index, dist, q, KnnResult(5))
    for c in cres
        @inbounds for i in index.lists[c.objID]
            d = dist(q, index.X[i])
            push!(res, i, d)
        end
    end

    res
end
