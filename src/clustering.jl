# This file is a part of KCenters.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

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
    
    dnet(callback, dist, X, floor(Int, n / numcenters))
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
    invindex = labelmap(y)
    m = length(invindex)
    centers = Vector{T}(undef, m)
    counts = zeros(Int, m)
    for (i, L) in sort!(collect(invindex))
        C = @view X[L]
        centers[i] = centroid(C)
        counts[i] = length(C)
    end

    distances = [dist(X[i], centers[y[i]]) for i in eachindex(X)]
    (centroids=centers, counts=counts, codes=y, distances=distances, err=sum(distances))
end

"""
    kcenters(dist::Function, X::AbstractVector{T}, k::Integer, centroid::Function=mean; initial=:fft, maxiters=0, tol=0.001, recall=1.0) where T
    kcenters(dist::Function, X::AbstractVector{T}, C::AbstractzVector{T}, centroid::Function=mean; maxiters=30, tol=0.001, recall=1.0) where T

Performs a kcenters clustering of `X` using `dist` as distance function and `centroid` to compute centroid objects.
It is based on the k-means algorithm yet using different algorithms as initial clusters.
    - `:fft` the _farthest first traversal_ selects a set of farthest points among them to serve as cluster seeds.
    - `:dnet` the _density net_ algorithm selects a set of points following the same distribution of the datasets; in contrast with a random selection, `:dnet` ensures that the selected points are not ``\\lfloor n/k \\rfloor`` nearest neighbors.
    - `:sfft` the `:fft` over a ``k + \\log n`` random sample
    - `:sdnet` the `:dnet` over a ``k + \\log n`` random sample
    - `:rand` selects the set of random points along the dataset.

If recall is 1.0 then an exhaustive search is made to find associations of each item to its nearest cluster; if ``0 < recall < 0`` then an approximate index
(`SearchGraph` from `SimilaritySearch.jl`) will be used for the same purpose; the `recall` controls the expected search quality (trade with search time).
"""
function kcenters(dist::Function, X::AbstractVector{T}, k::Integer, centroid::Function=mean; initial=:fft, maxiters=10, tol=0.001, recall=1.0, verbose=false) where T
    local err::Float64 = 0.0
    
    if initial == :fft
        m = 0
        irefs = enet(dist, X, k+m).irefs
        if m > 0
            irefs = irefs[1+m:end]
        end
        initial = X[irefs]
    elseif initial == :dnet
        irefs = dnet(dist, X, k).irefs
        resize!(irefs, k)
        initial = X[irefs]
    elseif initial == :sfft
        n = length(X)
        m = min(n, ceil(Int, sqrt(n)) + k)
        X_ = X[unique(rand(1:n, m))]
        C = enet(dist, X_, k, verbose=verbose)
        initial = X_[C.irefs]
    elseif initial == :sdnet
        n = length(X)
        m = min(n, ceil(Int, sqrt(n)) + k)
        X_ = X[unique(rand(1:n, m))]
        irefs = dnet(dist, X_, k, verbose=verbose).irefs
        resize!(irefs, k)
        initial = X_[irefs]
    elseif initial == :fftdensity
        n = length(X)
        m = min(n, ceil(Int, log(n)) + 2 * k)
        E = enet(dist, X, m)

        C = Dict{Int, Int}()
        # D = Dict{Int, Float64}()
        for p in E.seq
            i = first(p).objID
            C[i] = get(C, i, 0) + 1
            d = last(p).dist
            # D[i] = max(get(D, i, 0.0), d)
        end
        
        irefs = [s[1] for s in sort!(collect(C), by=p->p[2], rev=true)[1:2*k]]
        XX = X[irefs]
        C = enet(dist, XX, k)
        initial = XX[C.irefs]
    elseif initial == :rand
        initial = rand(X, k)
    elseif initial isa Symbol
        error("Unknown kind of initial value $initial")
    else
        initial = initial::AbstractVector{T}
    end

    kcenters(dist, X, initial, centroid, maxiters=maxiters, tol=tol, recall=recall, verbose=verbose)
end

function kcenters(dist::Function, X::AbstractVector{T}, C::AbstractVector{T}, centroid::Function=mean; maxiters=-1, tol=0.001, recall=1.0, verbose=true) where T
    # Lloyd's algoritm (kmeans)
    n = length(X)
    numcenters = length(C)
    if maxiters == -1
        maxiters = ceil(Int, log2(n))
    end

    function create_index(CC)
        if recall >= 1.0
            fit(Sequential, CC)
        else
            fit(SearchGraph, dist, CC, recall=recall)
        end
    end

    counts = zeros(Int, numcenters)
    codes = Vector{Int}(undef, n)
    distances = zeros(Float64, n)
    err = [typemax(Float64), associate_centroids_and_compute_error!(dist, X, create_index(C), codes, distances, counts)]
    iter = 0

    while iter < maxiters && err[end-1] - err[end] >= tol
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
        s = associate_centroids_and_compute_error!(dist, X, create_index(C), codes, distances, counts)

        push!(err, s)
        @assert !isnan(err[end]) "ERROR invalid score $err"
        verbose && println(stderr, "*** new score with $(numcenters) references: $err ***")
    end
    
    verbose && println(stderr, "*** finished computation of $(numcenters) references, err: $err ***")
    (centroids=C, counts=counts, codes=codes, distances=distances, err=err)
end

function associate_centroids_and_compute_error!(dist, X, index::Index, codes, distances, counters)
    res = KnnResult(1)
    for objID in eachindex(X)
        empty!(res)
        res = search(index, dist, X[objID], res)
        refID = first(res).objID
        codes[objID] = refID
        distances[objID] = last(res).dist
        counters[refID] += 1
    end

    mean(distances)
end

"""
    associate_centroids(dist, X, centers)

Returns the named tuple `(codes=codes, counts=counts, distances=distances, err=s)` where codes contains the nearest centroid
index for each item in `X` under the context of the `dist` distance function. `C` is the set of centroids and
`X` the dataset of objects. `C` also can be provided as a SimilaritySearch's Index.
"""
function associate_centroids(dist, X, centers)
    n = length(X)
    counts = zeros(Int, length(centers))
    codes = Vector{Int}(undef, n)
    distances = Vector{Float64}(undef, n)
    if centers isa AbstractVector
        centers = fit(Sequential, centers)
    end
    s = associate_centroids_and_compute_error!(dist, X, centers, codes, distances, counts)
    (codes=codes, distances=distances, err=s)
end

