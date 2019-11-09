import StatsBase: fit, predict
using StatsBase
export OneClassClassifier, fit, predict

mutable struct OneClassClassifier{T}
    centers::Vector{T}
    freqs::Vector{Int}
    n::Int
    dmax::Vector{Float64}
end

function fit(::Type{OneClassClassifier}, kcenters_::NamedTuple) where T
    k = length(kcenters_.centroids)
    freqs = zeros(Int, k)
    dmax = zeros(Float64, k)
    for i in eachindex(kcenters_.codes)
        code = kcenters_.codes[i]
        d = kcenters_.distances[i]
        freqs[code] += 1
        dmax[code] = max(dmax[code], d)
    end

    OneClassClassifier(kcenters_.centroids, freqs, length(kcenters_.codes), dmax)
end

# function regions(dist::Function, X, refs::Index)
#     I = invindex(dist, X, refs, k=1)
#     (freqs=[length(lst) for lst in I], regions=I)
# end

# function regions(dist::Function, X, refs)
#     regions(dist, X, fit(Sequential, refs))
# end

function predict(occ::OneClassClassifier{T}, dist::Function, q::T) where T
    seq = fit(Sequential, occ.centers)
    res = search(seq, dist, q, KnnResult(1))
    nn = first(res).objID
    (similarity=max(0.0, 1.0 - first(res).dist  / occ.dmax[nn]), freq=occ.freqs[nn])
    #occ.freqs[first(res).objID] / occ.n
end
