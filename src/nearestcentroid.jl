# based on Rocchio implementation (rocchio.jl) of https://github.com/sadit/TextSearch.jl

using SimilaritySearch
using LinearAlgebra
using StatsBase
import StatsBase: fit, predict
export NearestCentroid, fit, predict

"""
A simple nearest centroid classifier with support for kernel functions
"""
mutable struct NearestCentroid{T}
    centers::Vector{T}
    dmax::Vector{Float64}
    class_map::Vector{Int}
end

"""
    fit(::Type{NearestCentroid}, D::DeloneHistogram, class_map::Vector{Int}=Int[]; verbose=true)
    fit(::Type{NearestCentroid}, D::DeloneInvIndex, labels::AbstractVector; verbose=true)

Creates a NearestCentroid classifier using the output of either `kcenters` or `kcenters` as input
through either a `DeloneHistogram` or `DeloneInvIndex` struct.
If `class_map` is given, then it contains the list of labels to be reported associated to centers; if they are not specified,
then they are assigned in consecutive order for `DeloneHistogram` and as the most popular label for `DeloneInvIndex`.
"""
function fit(::Type{NearestCentroid}, D::DeloneHistogram, class_map::Vector{Int}=Int[]; verbose=true)
    if length(class_map) == 0
        class_map = collect(1:length(D.centers.db))
    end

    NearestCentroid(D.centers.db, D.dmax, class_map)
end

function fit(::Type{NearestCentroid}, D::DeloneInvIndex, labels::AbstractVector; verbose=false)
    m = length(D.lists)
    class_map = Vector{Int}(undef, m)
    nclasses = length(unique(labels))
    _ent2(f, n) = f == 0 ? 0.0 : f / n * log(n / f)

    for i in 1:m
        lst = D.lists[i]
        freqs = counts(labels[lst], 1:nclasses)
        if verbose
            n = sum(freqs)
            ent = sum(_ent2(f, n) for f in freqs) / log(nclasses)
            println(stderr, "centroid: $i, normalized-entropy: $ent, ", freqs)
        end
        freq, pos = findmax(freqs)
        class_map[i] = pos
    end

    NearestCentroid(D.centers.db, D.dmax, class_map)
end

"""
    predict(nc::NearestCentroid{T}, kernel::Function, x::T) where T
    predict(nc::NearestCentroid{T}, kernel::Function, X::AbstractVector{T}) where T

Predicts the class of `x` using the label of the nearest centroid under the `kernel` function
"""
function predict(nc::NearestCentroid{T}, kernel::Function, X::AbstractVector{T}) where T
    res = KnnResult(1)
    C = nc.centers
    dmax = nc.dmax
    L = Vector{Int}(undef, length(X))
    for j in eachindex(X)
        empty!(res)
        x = X[j]
        for i in eachindex(C)
            s = eval_kernel(kernel, x, C[i], dmax[i])
            push!(res, i, -s)
        end

        L[j] = nc.class_map[first(res).objID]
    end
    
    L
end

function predict(nc::NearestCentroid{T}, kernel::Function, x::T) where T
    predict(nc, kernel, [x])[1]
end

function eval_kernel(kernel, a, b, σ)
    kernel(a, b, σ)
end

function broadcastable(nc::NearestCentroid)
    (nc,)
end
