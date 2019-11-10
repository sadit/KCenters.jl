import StatsBase: fit, predict
using StatsBase
using SimilaritySearch
export VoronoiHistogram, fit, predict

mutable struct VoronoiHistogram
    centers::Index
    freqs::Vector{Int}
    dmax::Vector{Float64}
    n::Int
end

function fit(::Type{VoronoiHistogram}, kcenters_::NamedTuple) where T
    k = length(kcenters_.centroids)
    freqs = zeros(Int, k)
    dmax = zeros(Float64, k)
    
    for i in eachindex(kcenters_.codes)
        code = kcenters_.codes[i]
        d = kcenters_.distances[i]
        freqs[code] += 1
        dmax[code] = max(dmax[code], d)
    end

    C = fit(Sequential, kcenters_.centroids)
    VoronoiHistogram(C, freqs, dmax, length(kcenters_.codes))
end

function predict(occ::VoronoiHistogram, dist::Function, q)
    transform(occ, dist, q).sim > 0
end

function transform(occ::VoronoiHistogram, dist::Function, q::T) where T
    res = search(occ.centers, dist, q, KnnResult(1))
    i = first(res).objID
    sim = max(0.0, 1.0 - first(res).dist  / occ.dmax[i])
    (sim=sim, prob=occ.freqs[i] / occ.n)
end
