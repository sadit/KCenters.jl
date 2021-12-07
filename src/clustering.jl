# This file is a part of KCenters.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

using SimilaritySearch
using CategoricalArrays, StatsBase, MLDataUtils
export enet, dnet, kcenters, associate_centroids, ClusteringData

"""
    struct ClusteringData{DataType}
        # n elements in the dataset, m centers
        centers::DataType # centers, m entries
        freqs::Vector{Int32} # number of elements associated to each center, m entries
        dmax::Vector{Float32} # stores the distant element associated to each center, m entries
        codes::Vector{Int32} # id of the associated center, n entries
        distances::Vector{Float32} # from each element to its nearest center (label), n entries
        err::Vector{Float32} # dynamic of the error function, at least one entry
    end

The datastructure output of our clustering procedures
"""
struct ClusteringData{DataType<:AbstractDatabase}
    # n elements in the dataset, m centers
    centers::DataType # centers, m entries
    freqs::Vector{Int32} # number of elements associated to each center, m entries
    dmax::Vector{Float32} # stores the distant element associated to each center, m entries
    codes::Vector{Int32} # id of the associated center, n entries
    distances::Vector{Float32} # from each element to its nearest center (label), n entries
    err::Vector{Float32} # dynamic of the error function, at least one entry
end

"""
    kcenters(dist::PreMetric, X, y::CategoricalArray, sel::AbstractCenterSelection=CentroidSelection())

Computes a center per region (each region is defined by the set of items having the same label in `y`).
The output is compatible with `kcenters` function when `eltype(y)` is Int
"""
function kcenters(dist::PreMetric, X, y::CategoricalArray, sel::AbstractCenterSelection=CentroidSelection())
    X = convert(AbstractDatabase, X)
    m = length(levels(y))
    centers = Vector(undef, m)
    freqs = zeros(Int32, m)
    invindex = labelmap(y.refs)
    
    for i in 1:m
        elements = @view X[invindex[i]]
        centers[i] = center(sel, elements)
        freqs[i] = length(elements)
    end

    let centers = VectorDatabase(centers)
        distances = Float32[evaluate(dist, X[i], centers[y.refs[i]]) for i in eachindex(X)]
        codes = Int32.(y.refs)
        ClusteringData(centers, freqs, compute_dmax(m, codes, distances), codes, distances, Float32[sum(distances)])
    end
end

function compute_dmax(m, codes, distances)
    dmax = zeros(Float32, m)
    for i in eachindex(codes)
        code = codes[i]
        d = distances[i]
        dmax[code] = max(dmax[code], d)
    end
    dmax
end

"""
    kcenters(dist::PreMetric, X, k::Integer; sel::AbstractCenterSelection=CentroidSelection(), initial=:fft, maxiters=0, tol=0.001, recall=1.0)
    kcenters(dist::PreMetric, X, C; sel::AbstractCenterSelection=CentroidSelection(), maxiters=30, tol=0.001, recall=1.0)

Performs a kcenters clustering of `X` using `dist` as distance function and `sel` to compute center objects.
It is based on the Lloyd's algorithm yet using different algorithms as initial clusters:

- `:fft` the _farthest first traversal_ selects a set of farthest points among them to serve as cluster seeds.
- `:dnet` the _density net_ algorithm selects a set of points following the same distribution of the datasets; in contrast with a random selection, `:dnet` ensures that the selected points are not ``\\lfloor n/k \\rfloor`` nearest neighbors.
- `:sfft` the `:fft` over a ``k + \\log n`` random sample
- `:sdnet` the `:dnet` over a ``k + \\log n`` random sample
- `:rand` selects the set of random points along the dataset.

If recall is 1.0 then an exhaustive search is made to find associations of each item to its nearest cluster; if ``0 < recall < 0`` then an approximate index
(`SearchGraph` from `SimilaritySearch.jl`) will be used for the same purpose; the `recall` controls the expected search quality (trade with search time).
"""
function kcenters(dist::PreMetric, X, k::Integer; sel::AbstractCenterSelection=CentroidSelection(), initial=:fft, maxiters=10, tol=0.001, recall=1.0, verbose=false)
    X = convert(AbstractDatabase, X)

    if initial === :fft
        m = 0
        irefs = enet(dist, X, k+m).irefs
        if m > 0
            irefs = irefs[1+m:end]
        end
        initial = X[irefs]
    elseif initial === :dnet
        irefs = dnet(dist, X, k).irefs
        resize!(irefs, k)
        initial = X[irefs]
    elseif initial === :sfft
        n = length(X)
        m = min(n, ceil(Int, sqrt(n)) + k)
        X_ = X[unique(rand(1:n, m))]
        C = enet(dist, X_, k, verbose=verbose)
        initial = X_[C.irefs]
    elseif initial === :sdnet
        n = length(X)
        m = min(n, ceil(Int, sqrt(n)) + k)
        X_ = X[unique(rand(1:n, m))]
        irefs = dnet(dist, X_, k, verbose=verbose).irefs
        resize!(irefs, k)
        initial = X_[irefs]
    elseif initial === :fftdensity
        n = length(X)
        m = min(n, ceil(Int, log(n)) + 2 * k)

        E = enet(dist, X, m)

        C = Dict{Int, Int}()
        # D = Dict{Int, Float64}()
        for p in E.seq
            i = argmin(p)
            C[i] = get(C, i, 0) + 1
            # d = maximum(p)
            # D[i] = max(get(D, i, 0.0), d)
        end
        
        irefs = [s[1] for s in sort!(collect(C), by=p->p[2], rev=true)[1:2*k]]
        XX = X[irefs]
        C = enet(dist, XX, k)
        initial = XX[C.irefs]
    elseif initial === :rand
        initial = rand(X, k)
    elseif initial isa Symbol
        error("Unknown kind of initial value $initial")
    end

    kcenters_(dist, X, initial, sel=sel, maxiters=maxiters, tol=tol, recall=recall, verbose=verbose)
end

function kcenters_(dist::PreMetric, X::AbstractDatabase, C; sel::AbstractCenterSelection=CentroidSelection(), maxiters=-1, tol=0.001, recall=1.0, verbose=true)
    # Lloyd's algoritm
    n = length(X)
    numcenters = length(C)
    
    if maxiters == -1
        maxiters = ceil(Int, log2(n))
    end

    function create_index(CC)
        CC = convert(AbstractDatabase, CC)
        if recall >= 1.0
            ExhaustiveSearch(dist, CC)
        else
            idx = SearchGraph(; db=CC, dist)
            index!(idx)
        end
    end

    freqs = zeros(Int32, numcenters)
    codes = Vector{Int32}(undef, n)
    distances = zeros(Float32, n)
    err = Float32[typemax(Float32), associate_centroids_and_compute_error!(X, create_index(C), codes, distances, freqs)]
    iter = 0
    CC = C
    clusters = [Int[] for i in 1:numcenters]
    while iter < maxiters && err[end-1] - err[end] >= tol
        iter += 1
        verbose && println(stderr, "*** starting iteration: $iter; err: $err ***")
        for c in clusters
            empty!(c)
        end

        for (objID, plist) in enumerate(codes)
            for refID in plist
                push!(clusters[refID], objID)
            end
        end
        
        verbose && println(stderr, "*** computing centroids ***")
        resize!(CC, length(clusters))
        Threads.@threads for i in 1:length(clusters)
            plist = clusters[i]
            # CC[i] can be empty because we could be using approximate search
            if length(plist) > 0
                c = center(sel, X[plist])
                CC[i] = c
            end
        end
        
        verbose && println(stderr, "*** computing $(numcenters) nearest references ***")
        s = associate_centroids_and_compute_error!(X, create_index(CC), codes, distances, freqs)
        push!(err, s)
        isnan(err[end]) && error("ERROR invalid score $err")
        verbose && println(stderr, "*** new score with $(numcenters) references: $err ***")
    end
    
    verbose && println(stderr, "*** finished computation of $(numcenters) references, err: $err ***")
    ClusteringData(VectorDatabase(CC), freqs, compute_dmax(numcenters, codes, distances), codes, distances, err)
end

function associate_centroids_and_compute_error!(X, index::AbstractSearchContext, codes, distances, counters)
    Threads.@threads for objID in 1:length(X)
        res = KnnResult(1)
        search(index, X[objID], res)
        codes[objID] = argmin(res)
        distances[objID] = maximum(res)
    end

    for refID in codes 
        counters[refID] += 1
    end

    mean(distances)
end

"""
    associate_centroids(dist::PreMetric, X, centers)

Returns the named tuple `(codes=codes, freqs=freqs, distances=distances, err=s)` where codes contains the nearest center
index for each item in `X` under the context of the `dist` distance function. `C` is the set of centroids and
`X` the dataset of objects. `C` also can be provided as a SimilaritySearch's Index.
"""
function associate_centroids(dist::PreMetric, X, centers)
    n = length(X)
    freqs = zeros(Int, length(centers))
    codes = Vector{Int}(undef, n)
    distances = Vector{Float64}(undef, n)
    if centers isa AbstractVector
        centers = ExhaustiveSearch(dist, centers)
    end
    s = associate_centroids_and_compute_error!(X, centers, codes, distances, freqs)
    (codes=codes, distances=distances, err=s)
end

