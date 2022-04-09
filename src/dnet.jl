# This file is a part of KCenters.jl

using SimilaritySearch
using Random
export dnet

"""
    dnet(callback::Function, dist::SemiMetric, X::AbstractDatabase, k::Integer)

A `k`-net is a set of points `M` such that each object in `X` can be:
- It is in `M`
- It is in the knn set of an object in `M` (defined with the distance function `dist`)

The size of `M` is determined by \$\\leftceil |X| / k \\rightceil\$

The dnet function uses the `callback` function as an output mechanism. This function is called on each center as `callback(centerId, res, dbmap)` where
res is a `KnnResult` object (from SimilaritySearch.jl) and dbmap a mapping 

"""
function dnet(callback::Function, dist::SemiMetric, X::AbstractDatabase, k::Integer)
    N = length(X)
    S = SubDatabase(X, shuffle!(collect(1:N)))
    I = ParallelExhaustiveSearch(dist, S)
    res = KnnResult(k)
    rlist = Int32[]
    while length(I) > 0
        n = length(I)
        p = search(I, I[n], reuse!(res, k))
        callback(S.map[n], p.res, S.map)
        m = n - length(p.res)
        empty!(rlist)
        append!(rlist, idview(res))
        sort!(rlist)
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
    dnet(dist::SemiMetric, X::AbstractDatabase, numcenters::Integer)

Selects `numcenters` far from each other based on density nets.

- `dist` distance function
- `X` the objects to be computed
- `numcenters` number of centers to be computed

Returns a named tuple ``(nn, irefs, dmax)``.

- `irefs` contains the list of centers (indexes to ``X``)
- `seq` contains the ``k`` nearest references for each object in ``X`` (in ``X`` order) 
- `dmax` a list of coverage-radius of each center (aligned with irefs centers) smallest distance among centers

"""
function dnet(dist::SemiMetric, X::AbstractDatabase, numcenters::Integer; verbose=false)
    # criterion = change_criterion(0.01)
    n = length(X)
    irefs = Int32[]
    dmax = Float32[]
    seq = [KnnResult(1) for _ in 1:n]
    
    function callback(c, res, map)
        push!(irefs, c)
        push!(dmax, maximum(res))
        
        for (id, d) in res
            s = seq[map[id]]
            push!(s, c, d)
        end
    
        verbose && println(stderr, "dnet -- selected-center: $(length(irefs)), id: $c, dmax: $(dmax[end])")
    end
    

    dnet(callback, dist, X, n ÷ numcenters)
    #@info [length(p) for p in seq]
    #@info sort(irefs), sum([length(p) for p in seq]), length(irefs)
    (irefs=irefs, seq=seq, dmax=dmax)
end
