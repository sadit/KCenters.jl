using SimilaritySearch
import StatsBase: fit
using Random
export dnet


"""
    dnet(callback::Function, dist::Function, X::AbstractVector{T}, k::Int) where {T}

A `k`-net is a set of points `M` such that each object in `X` can be:
- It is in `M`
- It is in the knn set of an object in `M` (defined with the distance function `dist`)

The size of `M` is determined by ``\\leftceil|X|/k\\rightceil``

The dnet function uses the `callback` function as an output mechanism. This function is called on each center as `callback(centerId, res)` where
res is a `KnnResult` object (from SimilaritySearch.jl).

"""
function dnet(callback::Function, dist::Function, X::AbstractVector{T}, k::Int) where {T}
    N = length(X)
    metadist = (a::Int, b::Int) -> dist(X[a], X[b])
    I = fit(Sequential, shuffle!(collect(1:N)))
    res = KnnResult(k)

    while length(I.db) > 0
        empty!(res)
        n = length(I.db)
        search(I, metadist, n, res)
        callback(I.db[n], res, I.db)
        m = n - length(res)
        rlist = sort!([p.objID for p in res])
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
