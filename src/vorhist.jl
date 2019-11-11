import StatsBase: fit, predict
using StatsBase
using SimilaritySearch
export DeloneHistogram, fit, predict

mutable struct DeloneHistogram
    centers::Index
    freqs::Vector{Int}
    dmax::Vector{Float64}
    n::Int
end

function fit(::Type{DeloneHistogram}, kcenters_::NamedTuple) where T
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
    DeloneHistogram(C, freqs, dmax, length(kcenters_.codes))
end

function predict(vor::DeloneHistogram, dist::Function, q)
    res = search(vor.centers, dist, q, KnnResult(1))
    i = first(res).objID
    sim = max(0.0, 1.0 - first(res).dist  / vor.dmax[i])
    sim > 0
end
