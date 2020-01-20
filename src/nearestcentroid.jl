# This file is a part of KCenters.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt
# based on Rocchio implementation (rocchio.jl) of https://github.com/sadit/TextSearch.jl

using SimilaritySearch
using LinearAlgebra
using StatsBase
import StatsBase: fit, predict
export KNC, NearestCentroid, fit, predict, transform

"""
A simple nearest centroid classifier with support for kernel functions
"""
mutable struct KNC{T}
    centers::Vector{T}
    dmax::Vector{Float64}
    class_map::Vector{Int}
    nclasses::Int
end
const NearestCentroid = KNC

"""
    fit(::Type{KNC}, D::DeloneHistogram, class_map::Vector{Int}=Int[]; verbose=true)
    fit(::Type{KNC}, D::DeloneInvIndex, labels::AbstractVector; verbose=true)
    fit(::Type{KNC}, dist::Function, input_clusters::NamedTuple, train_X::AbstractVector, train_y::AbstractVector{_Integer}, centroid::Function=mean; split_entropy=0.1, verbose=false) where _Integer<:Integer

Creates a KNC classifier using the output of either `kcenters` or `kcenters` as input
through either a `DeloneHistogram` or `DeloneInvIndex` struct.
If `class_map` is given, then it contains the list of labels to be reported associated to centers; if they are not specified,
then they are assigned in consecutive order for `DeloneHistogram` and as the most popular label for `DeloneInvIndex`.

The third form is a little bit more complex, the idea is to divide clusters whenever their label-diversity surpasses a given threshold (measured with `split_entropy`).
This function receives a distance function `dist` and the original dataset `train_X` in addition to other mentioned arguments.
"""
function fit(::Type{KNC}, D::DeloneHistogram, class_map::Vector{Int}=Int[]; verbose=true)
    if length(class_map) == 0
        class_map = collect(1:length(D.centers.db))
    end

    KNC(D.centers.db, D.dmax, class_map, length(unique(class_map)))
end

function fit(::Type{KNC}, C::NamedTuple, class_map::Vector{Int}=Int[]; verbose=true)
    D = fit(DeloneHistogram, C)
    fit(KNC, D, class_map)
end

function fit(::Type{KNC},
        dist::Function, input_clusters::NamedTuple,
        train_X::AbstractVector,
        train_y::AbstractVector{_Integer},
        centroid::Function=mean;
        split_entropy=0.3,
        minimum_elements_per_centroid=1,
        verbose=false
    ) where _Integer<:Integer
    
    _lists = labelmap(input_clusters.codes)
    centroids = eltype(train_X)[] # clusters
    classes = Int[] # class mapping between clusters and classes
    dmax = Float64[]
    m = length(input_clusters.centroids)
    nclasses = length(unique(train_y))

    _ent(f, n) = (f == 0) ? 0.0 : (f / n * log(n / f))
    
    for i in 1:m
        lst = get(_lists, i, nothing)
        lst === nothing && continue
        #ylst = @view train_y[lst]
        ylst = train_y[lst]
        freqs = counts(ylst, 1:nclasses)
        labels = findall(f -> f >= minimum_elements_per_centroid, freqs)
        if length(labels) == 0
            verbose && println(stderr, "*** center $i: ignoring all elements because minimum-frequency restrictions were not met, freq >= $minimum_elements_per_centroid, freqs: $freqs")
            continue
        else
            verbose && println(stderr, "*** center $i: selecting labels $labels (freq >= $minimum_elements_per_centroid) [from $freqs]")
        end

        if length(labels) == 1
            freqs_ = freqs
            e = 0.0
        else
            freqs_ = freqs[labels]
            n = sum(freqs_)
            e = sum(_ent(f, n) for f in freqs_) / log(length(labels))
            verbose && println(stderr, "** centroid: $i, normalized-entropy: $e, ", 
                collect(zip(labels, freqs_)))
        end

        if e > split_entropy
            # X = @view train_X[lst]
            X = train_X[lst]
            invindex = labelmap(ylst)
            for l in labels
                XX = X[invindex[l]]
                c = centroid(XX)
                push!(centroids, c)
                push!(classes, l)
                d = 0.0
                for u in XX
                    d = max(d, dist(u, c))
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
    KNC(centroids, dmax, classes, nclasses)
end

function fit(::Type{KNC}, D::DeloneInvIndex, train_y::AbstractVector; verbose=false)
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

    KNC(D.centers.db, D.dmax, class_map, nclasses)
end

"""
    predict(nc::KNC{T}, kernel::Function, X::AbstractVector{T}, k=1) where T

Predicts the class of `x` using the label of the `k` nearest centroid under the `kernel` function.
"""
function predict(nc::KNC{T}, kernel::Function, X::AbstractVector{T}, k=1) where T
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

function predict(nc::KNC{T}, kernel::Function, x::T) where T
    predict(nc, kernel, [x])[1]
end


"""
    eval_kernel(kernel::Function, a, b, σ)

Evaluates a kernel function over the giver arguments (isolated to ensure that the function can be compiled)
"""
function eval_kernel(kernel::Function, a, b, σ)
    kernel(a, b, σ)
end

function broadcastable(nc::KNC)
    (nc,)
end
