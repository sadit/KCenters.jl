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
    nclasses::Int
end

"""
    fit(::Type{NearestCentroid}, D::DeloneHistogram, class_map::Vector{Int}=Int[]; verbose=true)
    fit(::Type{NearestCentroid}, D::DeloneInvIndex, labels::AbstractVector; verbose=true)
    fit(::Type{NearestCentroid}, dist::Function, input_clusters::NamedTuple, train_X::AbstractVector, train_y::AbstractVector{_Integer}, centroid::Function=mean; split_entropy=0.1, verbose=false) where _Integer<:Integer

Creates a NearestCentroid classifier using the output of either `kcenters` or `kcenters` as input
through either a `DeloneHistogram` or `DeloneInvIndex` struct.
If `class_map` is given, then it contains the list of labels to be reported associated to centers; if they are not specified,
then they are assigned in consecutive order for `DeloneHistogram` and as the most popular label for `DeloneInvIndex`.

The third form is a little bit more complex, the idea is to divide clusters whenever their label-diversity surpasses a given threshold (measured with `split_entropy`).
This function receives a distance function `dist` and the original dataset `train_X` in addition to other mentioned arguments.
"""
function fit(::Type{NearestCentroid}, D::DeloneHistogram, class_map::Vector{Int}=Int[]; verbose=true)
    if length(class_map) == 0
        class_map = collect(1:length(D.centers.db))
    end

    NearestCentroid(D.centers.db, D.dmax, class_map, length(unique(class_map)))
end

function fit(::Type{NearestCentroid}, C::NamedTuple, class_map::Vector{Int}=Int[]; verbose=true)
    D = fit(DeloneHistogram, C)
    fit(NearestCentroid, D, class_map)
end

function _labelmap(codes)
    _lists = Dict{Int,Vector{Int}}()
    for (pos, code) in enumerate(codes)
        lst = get(_lists, code, nothing)
        if lst === nothing
            _lists[code] = [pos]
        else
            push!(lst, pos)
        end
    end

    _lists
end

function fit(::Type{NearestCentroid}, dist::Function, input_clusters::NamedTuple, train_X::AbstractVector, train_y::AbstractVector{_Integer}, centroid::Function=mean; split_entropy=0.3, verbose=false) where _Integer<:Integer
    _lists = _labelmap(input_clusters.codes)
    centroids = eltype(train_X)[] # clusters
    classes = Int[] # class mapping between clusters and classes
    dmax = Float64[]
    m = length(input_clusters.centroids)
    nclasses = length(unique(train_y))
    
    _ent2(f, n) = (f == 0) ? 0.0 : (f / n * log(n / f))
    for i in 1:m
        lst = get(_lists, i, nothing)
        lst === nothing && continue
        freqs = counts(train_y[lst], 1:nclasses)
        labels = findall(f -> f > 0, freqs)
        e = Float64(length(labels))
        if e == 1.0
            e = 0.0
        else
            n = sum(freqs)
            invlognclasses = 1 / log(length(labels))
            e = sum(_ent2(f, n) for f in freqs) * invlognclasses
        end

        verbose && println(stderr, "** centroid: $i, normalized-entropy: $e, ", freqs)
        if e > split_entropy
            labels = findall(f -> f > 0, freqs)
            L = train_y[lst]
            verbose && println(stderr, "centroid $i, labels=$labels, freqs: $freqs")
            for (j, l) in enumerate(labels)
                LL = lst[L .== l]
                c = centroid(train_X[LL])
                push!(centroids, c)
                push!(classes, l)
                d = 0.0
                for objID in LL
                    d = max(d, dist(train_X[objID], c))
                end
                push!(dmax, d)
            end
        else
            push!(centroids, input_clusters.centroids[i])
            freq, pos = findmax(freqs)
            push!(classes, pos)
            d = 0.0
            for objID in lst
                d = max(d, dist(train_X[objID], centroids[end]))
            end
            push!(dmax, d)
         end
    end

    verbose && println(stderr, "finished with $(length(centroids)) centroids; started with $(length(input_clusters.centroids))")
    NearestCentroid(centroids, dmax, classes, nclasses)
end

function fit(::Type{NearestCentroid}, D::DeloneInvIndex, train_y::AbstractVector; verbose=false)
    m = length(D.lists)
    class_map = Vector{Int}(undef, m)
    nclasses = length(unique(train_y))
    _ent2(f, n) = f == 0 ? 0.0 : f / n * log(n / f)

    for i in 1:m
        lst = D.lists[i]
        freqs = counts(train_y[lst], 1:nclasses)
        if verbose
            n = sum(freqs)
            ent = sum(_ent2(f, n) for f in freqs) / log(nclasses)
            println(stderr, "centroid: $i, normalized-entropy: $ent, ", freqs)
        end
        freq, pos = findmax(freqs)
        class_map[i] = pos
    end

    NearestCentroid(D.centers.db, D.dmax, class_map, nclasses)
end

"""
    predict(nc::NearestCentroid{T}, kernel::Function, x::T) where T
    predict(nc::NearestCentroid{T}, kernel::Function, X::AbstractVector{T}) where T

Predicts the class of `x` using the label of the nearest centroid under the `kernel` function
"""
function predict(nc::NearestCentroid{T}, kernel::Function, X::AbstractVector{T}, ::Type) where T
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

function predict(nc::NearestCentroid{T}, kernel::Function, X::AbstractVector{T}, k=1) where T
    res = KnnResult(k)
    C = nc.centers
    dmax = nc.dmax
    ypred = Vector{Int}(undef, length(X))
    for j in eachindex(X)
        empty!(res)
        x = X[j]
        for i in eachindex(C)
            s = eval_kernel(kernel, x, C[i], dmax[i])
            push!(res, i, -s)
        end

        c = counts([nc.class_map[p.objID] for p in res], 1:nc.nclasses)
        ypred[j] = findmax(c)[end]
    end
    
    ypred
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
