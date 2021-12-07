# This file is a part of KCenters.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

using SimilaritySearch
using Random
export dnet

"""
    dnet(callback::Function, dist::PreMetric, X::AbstractDatabase, k::Integer)

A `k`-net is a set of points `M` such that each object in `X` can be:
- It is in `M`
- It is in the knn set of an object in `M` (defined with the distance function `dist`)

The size of `M` is determined by \$\\leftceil |X| / k \\rightceil\$

The dnet function uses the `callback` function as an output mechanism. This function is called on each center as `callback(centerId, res, dbmap)` where
res is a `KnnResult` object (from SimilaritySearch.jl) and dbmap a mapping 

"""
function dnet(callback::Function, dist::PreMetric, X::AbstractDatabase, k::Integer)
    N = length(X)
    S = SubDatabase(X, shuffle!(collect(1:N)))
    I = ExhaustiveSearch(dist, S)
    res = KnnResult(k)

    while length(I) > 0
        empty!(res)
        n = length(I)
        search(I, I[n], res)
        callback(S.map[n], res, S.map)
        m = n - length(res)
        rlist = sort!([id_ for (id_, dist_) in res])
        numzeros = 0
        while length(rlist) > 0
            if rlist[end] > m
                S.map[rlist[end]] = 0
                pop!(rlist)
                numzeros += 1
            else
                break
            end
        end

        E = @view S.map[m+1:end]
        sort!(E)
        E = @view S.map[m+1+numzeros:end]
        if length(E) > 0
            S.map[rlist] .= E
        end

        resize!(S.map, m)
    end
end


"""
    dnet(dist::PreMetric, X::AbstractDatabase, numcenters::Integer)

Selects `numcenters` far from each other based on density nets.

- `dist` distance function
- `X` the objects to be computed
- `numcenters` number of centers to be computed

Returns a named tuple ``(nn, irefs, dmax)``.

- `irefs` contains the list of centers (indexes to ``X``)
- `seq` contains the ``k`` nearest references for each object in ``X`` (in ``X`` order) 
- `dmax` a list of coverage-radius of each center (aligned with irefs centers) smallest distance among centers

"""
function dnet(dist::PreMetric, X::AbstractDatabase, numcenters::Integer; verbose=false)
    # criterion = change_criterion(0.01)
    n = length(X)
    irefs = Int32[]
    dmax = Float32[]
    seq = [KnnResult(1) for i in 1:n]

    function callback(c, res, map)
        push!(irefs, c)
        push!(dmax, maximum(res))
        for (id, d) in res
            push!(seq[map[id]], c, d)
        end
    
       verbose && println(stderr, "dnet -- selected-center: $(length(irefs)), id: $c, dmax: $(dmax[end])")
    end
    
    dnet(callback, dist, X, floor(Int32, n / numcenters))
    #@info [length(p) for p in seq]
    #@info sort(irefs), sum([length(p) for p in seq]), length(irefs)
    (irefs=irefs, seq=seq, dmax=dmax)
end
