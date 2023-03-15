export references

"""
    references(
        weighting_centers::Function,
        dist::SemiMetric, db::AbstractDatabase, k::Integer;
        Δ=1.5,
        sample=Δ*k + sqrt(length(db)),
        maxiters=0,
        tol=0.001,
        initial=:rand)
    references(dist::SemiMetric, db::AbstractDatabase, k::Integer; kwargs...)

Computes a set of `k` references from `db`, see the [`kcenters`](@ref) documentation.

More precisely, the references will be computed from a sample subset (`sample`);
it computes Δ k references and select the `k` elements using the best ones w.r.t. `weighting_centers(C, i)` function
(where `C` is a `ClusteringData` object and `i` the i-th center).
The set of references are meaninful under `dist` metric function but also may follow some
characteristics given by the `initial` selection strategy.

# Arguments

- `dist`: a distance function
- `db`: the database to be sampled
- `k`: the number of centers to compute

# Keyword arguments
- `sample::Real`: indicates the sampling size before computing the set of `k` references, defaults to `log(|db|) k`; `sample=0` means for no sampling.
- `Δ::Real`: expands the number of candidates to be selected as references
- `maxiters::Int`: number of iterationso  of the Lloyd algorithm that should be applied on the initial computation of centers, that is, `maxiters > 0` applies `maxiters` iterations of the algorithm.
- `tol::Float64`: change tolerance to stop the Lloyd algorithm (error changes smaller than `tol` among iterations will stop the algorithm)
- `initial`: initial centers or a strategy to compute initial centers, symbols `:rand`, `:fft`, and `:dnet`.
There are several interactions between initial values and parameter interactions (described in `KCenters` object), for instance,
the `maxiters > 0` apply the Lloyd's algorithm to the initial computation of references.

- if `initial=:rand`:
  - `maxiters = 0` will retrieve a simple random sampling
  - `maxiters > 0' achieve kmeans-centroids, `maxiters` should be set appropiately for the the dataset
- if `initial=:dnet`:
  - `maxiters = 0` computes a pure density-net
  - `maxiters > 0` will compute a kmeans centroids but with an initialization based on the dnet
- if `initial=:fft`:
  - `maxiters = 0` computes `k` centers with the farthest first traversal algorithm
  - `maxiters > 0` will use the FFT based kcenters as initial points for the Lloyd algorithm

Note 1: `maxiters > 0` needs to compute centroids and these centroids should be _defined_
for the specific data model, and also be of use in the specific metric distance and error function.

Note 2: The error function is defined as the mean of distances of all objects in `db` to its associated nearest centers in each iteration.

Note 3: The computation of references on large databases can be prohibitive, in these cases please consider to work on a sample of `db`
"""
function references(
        weighting_centers::Function,
        dist::SemiMetric, db::AbstractDatabase, k::Integer;
        Δ=1.5,
        sample=Δ*k + sqrt(length(db)),
        maxiters=0,
        tol=0.001,
        initial=:rand)

    n = length(db)
    if n == k
        sample = 0
        Δk = k
    else
        sample = ceil(Int, sample)
        Δk = ceil(Int, Δ * k)
    end

    0 < k <= n || throw(ArgumentError("invalid relation between k and n, must follow 0 < k <= n"))
    C = if sample > 0
        s = unique(rand(eachindex(db), sample))
        kcenters(dist, SubDatabase(db, s), Δk; initial, maxiters, tol)
    else
        kcenters(dist, db, Δk; initial, maxiters, tol)
    end
    
    W = [weighting_centers(C, i) for i in eachindex(C.centers)]
    P = sortperm(W; rev=true)
    C.centers[P[1:k]]
end

function references(dist::SemiMetric, db::AbstractDatabase, k::Integer; kwargs...)
    references(dist, db, k; kwargs...) do C, i
        f = C.freqs[i]
        f * sign(C.dmax[i])
    end
end
