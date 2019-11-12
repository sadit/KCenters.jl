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
    fit(::Type{NearestCentroid}, D::DeloneHistogram, class_map::Vector{Int}=Int[])
    fit(::Type{NearestCentroid}, D::DeloneInvIndex, labels::AbstractVector)

Creates a NearestCentroid classifier using the output of either `kcenters` or `kcenters` as input
through either a `DeloneHistogram` or `DeloneInvIndex` struct.
If `class_map` is given, then it contains the list of labels to be reported associated to centers; if they are not specified,
then they are assigned in consecutive order for `DeloneHistogram` and as the most popular label for `DeloneInvIndex`.
"""
function fit(::Type{NearestCentroid}, D::DeloneHistogram, class_map::Vector{Int}=Int[])
    if length(class_map) == 0
        class_map = collect(1:length(D.centers.db))
    end

    NearestCentroid(D.centers.db, D.dmax, class_map)
end

function fit(::Type{NearestCentroid}, D::DeloneInvIndex, labels::AbstractVector)
    m = length(D.lists)
    class_map = Vector{Int}(undef, m)
    for i in 1:m
        lst = D.lists[i]
        freq, pos = findmax(counts(labels[lst], 1:m))
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
