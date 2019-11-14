# Copyright 2017-2019 Eric S. Tellez
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

using SimilaritySearch
using StatsBase
export enet, dnet, kcenters, kcenters, associate_centroids

"""
    enet(dist::Function, X::AbstractVector{T}, numcenters::Int, knr::Int=1; verbose=false) where T

Selects `numcenters` far from each other based on Farthest First Traversal.

- `dist` distance function
- `X` the objects to be computed
- `numcenters` number of centers to be computed
- `knr` number of nearest references per object (knr=1 defines a partition)

Returns a named tuple ``(nn, irefs, dmax)``.

- `irefs` contains the list of centers (indexes to ``X``)
- `seq` contains the ``k`` nearest references for each object in ``X`` (in ``X`` order) 
- `dmax` smallest distance among centers

"""
function enet(dist::Function, X::AbstractVector{T}, numcenters::Int, knr::Int=1; verbose=false) where T
    # refs = Vector{Float64}[]
    irefs = Int[]
    nn = [KnnResult(knr) for i in 1:length(X)]
    dmax = 0.0

    function callback(c, _dmax)
        push!(irefs, c)
        dmax = _dmax
        verbose && println(stderr, "computing fartest point $(length(irefs)), dmax: $dmax, imax: $c")
    end

    function capturenn(i, refID, d)
        push!(nn[i], refID, d)
    end

    fftraversal(callback, dist, X, size_criterion(numcenters), capturenn)
    return (irefs=irefs, seq=nn, dmax=dmax)
end

"""
    dnet(dist::Function, X::AbstractVector{T}, numcenters::Int, knr::Int) where T

Selects `numcenters` far from each other based on density nets.

- `dist` distance function
- `X` the objects to be computed
- `numcenters` number of centers to be computed

Returns a named tuple ``(nn, irefs, dmax)``.

- `irefs` contains the list of centers (indexes to ``X``)
- `seq` contains the ``k`` nearest references for each object in ``X`` (in ``X`` order) 
- `dmax` a list of coverage-radius of each center (aligned with irefs centers) smallest distance among centers

"""
function dnet(dist::Function, X::AbstractVector{T}, numcenters::Int; verbose=false) where T
    # criterion = change_criterion(0.01)
    n = length(X)
    irefs = Int[]
    dmax = Float64[]
    seq = [KnnResult(1) for i in 1:n]

    function callback(c, res, map)
        push!(irefs, c)
        push!(dmax, last(res).dist)
        for p in res
            push!(seq[map[p.objID]], c, p.dist)
        end
    
       verbose && println(stderr, "dnet -- selected-center: $(length(irefs)), id: $c, dmax: $(dmax[end])")
    end
    
    dnet(callback, dist, X, ceil(Int, n / numcenters))
    #@info [length(p) for p in seq]
    #@info sort(irefs), sum([length(p) for p in seq]), length(irefs)
    (irefs=irefs, seq=seq, dmax=dmax)
end

"""
    kcenters(dist::Function, X::AbstractVector{T}, y::AbstractVector, centroid::Function=mean) where T

Computes a centroid per region (each region is defined by the set of items having the same label in `y`).
The output is compatible with `kcenters` function when `eltype(y)` is Int
"""
function kcenters(dist::Function, X::AbstractVector{T}, y::AbstractVector{I}, centroid::Function=mean) where {T,I<:Integer}
    labels = sort!(unique(y))
    m = length(labels)
    centers = Vector{T}(undef, m)
    populations = Vector{Int}(undef, m)

    for i in eachindex(labels)
        label = labels[i]
        L = X[label .== y]
        centers[i] = centroid(L)
        populations[i] = length(L)
    end

    distances = [dist(X[i], centers[y[i]]) for i in eachindex(X)]
    (centroids=centers, codes=y, distances=distances, err=sum(distances))
end

"""
    kcenters(dist::Function, X::AbstractVector{T}, k::Integer, centroid::Function=mean; initial=:fft, maxiters=0, tol=0.001, recall=1.0) where T
    kcenters(dist::Function, X::AbstractVector{T}, C::AbstractVector{T}, centroid::Function=mean; maxiters=0, tol=0.001, recall=1.0) where T

Performs a kcenters clustering of `X` using `dist` as distance function and `centroid` to compute centroid objects.
It is based on the k-means algorithm yet using different algorithms as initial clusters.
If recall is 1.0 then an exhaustive search is made to find associations of each item to its nearest cluster; if ``0 < recall < 0`` then an approximate index
(`SearchGraph` from `SimilaritySearch.jl`) will be used for the same purpose; the `recall` controls the expected search quality (trade with search time).
"""
function kcenters(dist::Function, X::AbstractVector{T}, k::Integer, centroid::Function=mean; initial=:fft, maxiters=0, tol=0.001, recall=1.0, verbose=false) where T
    local err::Float64 = 0.0

    if initial in (:fft, :minmax, :enet)
        initial = X[enet(dist, X, k).irefs]
    elseif initial in (:dnet, :knnballs)
        initial = X[dnet(dist, X, k).irefs]
    elseif initial in (:rand, :random)
        initial = rand(X, k)
    else
        initial = initial::AbstractVector{T}
    end

    kcenters(dist, X, initial, centroid, maxiters=maxiters, tol=tol, recall=recall, verbose=verbose)
end

function kcenters(dist::Function, X::AbstractVector{T}, C::AbstractVector{T}, centroid::Function=mean; maxiters=0, tol=0.001, recall=1.0, verbose=false) where T
    # Lloyd's algoritm (kmeans)
    n = length(X)
    numcenters = length(C)
    if maxiters == 0
        maxiters = ceil(Int, sqrt(n))
    end

    function create_index(CC)
        if recall >= 1.0
            fit(Sequential, CC)
        else
            fit(SearchGraph, dist, CC, recall=recall)
        end
    end

    codes = Vector{Int}(undef, n)
    distances = zeros(Float64, n)
    err = [typemax(Float64), associate_centroids_and_compute_error(dist, create_index(C), X, codes, distances)]
    iter = 0

    while iter < maxiters && abs(err[end-1] - err[end]) >= tol
        iter += 1
        verbose && println(stderr, "*** starting iteration: $iter; err: $err ***")
        clusters = [Int[] for i in 1:numcenters]
        for (objID, plist) in enumerate(codes)
            for refID in plist
                push!(clusters[refID], objID)
            end
        end
        
        verbose && println(stderr, "*** computing centroids ***")
        for i in 1:length(clusters)
            plist = clusters[i]
            # C[i] can be empty because we could be using approximate search
            if length(plist) > 0
                C[i] = centroid(X[plist])
            end
        end
        
        verbose && println(stderr, "*** computing $(numcenters) nearest references ***")
        s = associate_centroids_and_compute_error(dist, create_index(C), X, codes, distances)

        push!(err, s)
        @assert !isnan(err[end]) "ERROR invalid score $err"
        verbose && println(stderr, "*** new score with $(numcenters) references: $err ***")
    end
    
    verbose && println(stderr, "*** finished computation of $(numcenters) references, err: $err ***")
    (centroids=C, codes=codes, distances=distances, err=err)
end

function associate_centroids_and_compute_error(dist, Cindex::Index, X, codes, distances)
    res = KnnResult(1)
    for objID in 1:length(X)
        empty!(res)
        res = search(Cindex, dist, X[objID], res)
        codes[objID] = first(res).objID
        distances[objID] = last(res).dist
    end

    sum(d^2 for d in distances) / length(distances)
    #mean(distances)
end

"""
    associate_centroids(dist, C, X)

Returns the named tuple `(codes=codes, distances=distances, err=s)` where codes contains the nearest centroid
index for each item in `X` under the context of the `dist` distance function. `C` is the set of centroids and
`X` the dataset of objects. `C` also can be provided as a SimilaritySearch's Index.
"""
function associate_centroids(dist, C, X)
    n = length(X)
    codes = Vector{Int}(undef, n)
    distances = Vector{Float64}(undef, n)
    if C isa AbstractVector
        C = fit(Sequential, C)
    end
    s = associate_centroids_and_compute_error(dist, C, X, codes, distances)
    (codes=codes, distances=distances, err=s)
end


export sum_intracluster_squared_distances, sum_intracluster_distances, mean_intracluster_squared_distances, mean_intracluster_distances, inertia

function sum_intracluster_squared_distances(nndist::AbstractVector)
    s = 0.0
    for d in nndist
        s += d^2
    end

    s
end

function sum_intracluster_distances(nndist::AbstractVector)
    sum(nndist)
end

mean_intracluster_squared_distances(nndist) = sum_intracluster_squared_distances(nndist) / length(nndist)
mean_intracluster_distances(nndist) = sum_intracluster_distances(nndist) / length(nndist)
inertia(nndist) = sum_intracluster_squared_distances(nndist)
