# This file is a part of KCenters.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

using SimilaritySearch
using Random
export dnet

struct MaskedDistance{DataType<:AbstractVector,DistType<:PreMetric} <: PreMetric
    dist::DistType
    db::DataType
end

SimilaritySearch.evaluate(m::MaskedDistance, i::Integer, j::Integer) = @inbounds evaluate(m.dist, m.db[i], m.db[j])

"""
    dnet(callback::Function, dist::PreMetric, X::AbstractVector{T}, k::Integer) where {T}

A `k`-net is a set of points `M` such that each object in `X` can be:
- It is in `M`
- It is in the knn set of an object in `M` (defined with the distance function `dist`)

The size of `M` is determined by \$\\leftceil |X| / k \\rightceil\$

The dnet function uses the `callback` function as an output mechanism. This function is called on each center as `callback(centerId, res)` where
res is a `KnnResult` object (from SimilaritySearch.jl).

"""
function dnet(callback::Function, dist::PreMetric, X::AbstractVector{T}, k::Integer) where {T}
    N = length(X)
    metadist = (a::Int, b::Int) -> evaluate(dist, X[a], X[b])

    I = ExhaustiveSearch(MaskedDistance(dist, X), shuffle!(collect(1:N)))
    res = KnnResult(k)

    while length(I.db) > 0
        empty!(res)
        n = length(I.db)
        search(I, n, res)
        callback(I.db[n], res, I.db)
        m = n - length(res)
        rlist = sort!([p.id for p in res])
        numzeros = 0
        while length(rlist) > 0
            if rlist[end] > m
                I.db[rlist[end]] = 0
                pop!(rlist)
                numzeros += 1
            else
                break
            end
        end

        E = @view I.db[m+1:end]
        sort!(E)
        E = @view I.db[m+1+numzeros:end]
        if length(E) > 0
            I.db[rlist] .= E
        end

        resize!(I.db, m)
    end
end


"""
    dnet(dist::PreMetric, X::AbstractVector{T}, numcenters::Integer) where T

Selects `numcenters` far from each other based on density nets.

- `dist` distance function
- `X` the objects to be computed
- `numcenters` number of centers to be computed

Returns a named tuple ``(nn, irefs, dmax)``.

- `irefs` contains the list of centers (indexes to ``X``)
- `seq` contains the ``k`` nearest references for each object in ``X`` (in ``X`` order) 
- `dmax` a list of coverage-radius of each center (aligned with irefs centers) smallest distance among centers

"""
function dnet(dist::PreMetric, X::AbstractVector{T}, numcenters::Integer; verbose=false) where T
    # criterion = change_criterion(0.01)
    n = length(X)
    irefs = Int32[]
    dmax = Float32[]
    seq = [KnnResult(1) for i in 1:n]

    function callback(c, res, map)
        push!(irefs, c)
        push!(dmax, last(res).dist)
        for p in res
            push!(seq[map[p.id]], c, p.dist)
        end
    
       verbose && println(stderr, "dnet -- selected-center: $(length(irefs)), id: $c, dmax: $(dmax[end])")
    end
    
    dnet(callback, dist, X, floor(Int32, n / numcenters))
    #@info [length(p) for p in seq]
    #@info sort(irefs), sum([length(p) for p in seq]), length(irefs)
    (irefs=irefs, seq=seq, dmax=dmax)
end
