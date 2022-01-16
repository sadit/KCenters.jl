# This file is a part of KCenters.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

using SimilaritySearch
export fftraversal, enet
using Distributed

function _ignore3(a, b, c)
end

"""
    fftraversal(callback::Function, dist::SemiMetric, X::AbstractDatabase, stop, callbackdist=_ignore3)

Selects a number of farthest points in `X`, using a farthest first traversal

- The callback function is called on each selected far point as `callback(centerID, dmax)` where `dmax` is the distance to the nearest previous reported center (the first is reported with typemax)
- The selected objects are far under the `dist` distance function with signature (T, T) -> Float64
- The number of points is determined by the stop criterion function with signature (Float64[], T[]) -> Bool
    - The first argument corresponds to the list of known distances (far objects)
    - The second argument corresponds to the database
- Check `criterions.jl` for basic implementations of stop criterions
- The callbackdist function is called on each distance evaluation between pivots and items in the dataset
    `callbackdist(index-pivot, index-item, distance)`
"""
function fftraversal(callback::Function, dist::SemiMetric, X::AbstractDatabase, stop, callbackdist=_ignore3)
    N = length(X)
    D = Vector{Float64}(undef, N)
    dmaxlist = Float64[]
    dset = [typemax(Float64) for i in 1:N]
    imax::Int = rand(1:N)
    dmax::Float64 = typemax(Float64)
    N == 0 && return
    k::Int = 0
    
    @inbounds while k <= N
        k += 1
        pivot = X[imax]
        push!(dmaxlist, dmax)
        callback(imax, dmax)
        dmax = 0.0
        ipivot = imax
        imax = 0

        D .= 0.0

        if nworkers() == 1
            Threads.@threads for i in 1:N
                D[i] = evaluate(dist, X[i], pivot)
            end
        else
            # only worths for very large datasets or very expensive distance functions
            D = pmap(obj -> evaluate(dist, obj, pivot), X)
        end

        for i in 1:N
            # d = evaluate(dist, X[i], pivot) 
            d = D[i]
            callbackdist(i, ipivot, d)

            if d < dset[i]
                dset[i] = d
            end

            if dset[i] > dmax
                dmax = dset[i]
                imax = i
            end
        end

        if dmax == 0 || stop(dmaxlist, X)
            break
        end
    end
end


"""
    enet(dist::SemiMetric, X::AbstractDatabase, numcenters::Int, knr::Int=1; verbose=false) where T

Selects `numcenters` far from each other based on Farthest First Traversal.

- `dist` distance function
- `X` the objects to be computed
- `numcenters` number of centers to be computed
- `knr` number of nearest references per object (knr=1 defines a partition)

Returns a named tuple \$(nn, irefs, dmax)\$.

- `irefs` contains the list of centers (indexes to ``X``)
- `seq` contains the ``k`` nearest references for each object in ``X`` (in ``X`` order) 
- `dmax` smallest distance among centers

"""
function enet(dist::SemiMetric, X::AbstractDatabase, numcenters::Integer, knr::Integer=1; verbose=false) where T
    # refs = Vector{Float64}[]
    irefs = Int32[]
    nn = [KnnResult(knr) for _ in 1:length(X)]
    dmax = zero(Float32)

    function callback(c, _dmax)
        push!(irefs, c)
        dmax = convert(Float32, _dmax)
        verbose && println(stderr, "computing fartest point $(length(irefs)), dmax: $dmax, imax: $c")
    end

    function capturenn(i, refID, d)
        @inbounds res = nn[i]
        @inbounds push!(res, refID, d)
    end

    fftraversal(callback, dist, X, size_criterion(numcenters), capturenn)
    return (irefs=irefs, seq=nn, dmax=dmax)
end
