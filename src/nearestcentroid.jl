# based on Rocchio implementation (rocchio.jl) of https://github.com/sadit/TextSearch.jl

using SimilaritySearch
using LinearAlgebra
using StatsBase: countmap
import StatsBase: fit, predict
export NearestCentroid, fit, predict

"""
A simple nearest centroid classifier with support for kernel functions
"""
mutable struct NearestCentroid{T}
    delone::DeloneHistogram{T}
end

"""
    fit(::Type{NearestCentroid}, delone::DeloneHistogram)

Creates a NearestCentroid classifier using the output of either `kcenters` or `kcenters_by_label` as input
through a `DeloneHistogram` struct
"""
function fit(::Type{NearestCentroid}, delone::DeloneHistogram)
    NearestCentroid(delone)
end

"""
    predict(nc::NearestCentroid{T}, kernel::Function, x::T) where T
    predict(nc::NearestCentroid{T}, kernel::Function, X::AbstractVector{T}) where T

Predicts the class of `x` using the label of the nearest centroid under the `kernel` function
"""
function predict(nc::NearestCentroid{T}, kernel::Function, X::AbstractVector{T}) where T
    res = KnnResult(1)
    C = nc.delone.centers.db
    dmax = nc.delone.dmax
    L = Vector{Int}(undef, length(X))
    for j in eachindex(X)
        empty!(res)
        x = X[j]
        for i in eachindex(C)
            s = eval_kernel(kernel, x, C[i], dmax[i])
            push!(res, i, -s)
        end

        L[j] = first(res).objID
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
