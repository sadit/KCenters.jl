# This file is a part of KCenters.jl

export partition, knr, sequence, invindex

"""
    partition(callback::Function, objects::AbstractVector{T}, refs::AbstractSearchIndex; k::Int=1) where T

Groups items in `objects` using a nearest neighbor rule over `refs`.
The output is controlled using a callback function. The call is performed in `objects` order.

- `callback` is a function that is called for each `(objID, refItem)`
- `objects` is the input dataset
- `dist` a distance function ``(T, T) \\rightarrow \\mathbb{R}``
- `refs` the list of references
- `k` specifies the number of nearest neighbors to use
- `indexclass` specifies the kind of index to be used, a function receiving `(refs, dist)` as arguments,
    and returning the new metric index

Please note that each object can be related to more than one group ``k > 1`` (default ``k=1``).

"""
function partition(callback::Function, objects::AbstractVector{T}, refs::AbstractSearchIndex; k::Int=1) where T
    res = KnnResult(k)
    for i in 1:length(objects)
        empty!(res)
        callback(i, search(refs, objects[i], res))
    end
end

"""
    invindex(objects::AbstractVector{T}, refs::AbstractSearchIndex; k::Int=1) where T

Creates an inverted index from references to objects.
So, an object ``u`` is in ``r``'s posting list iff ``r``
is among the ``k`` nearest references of ``u``.

"""
function invindex(objects::AbstractVector{T}, refs::AbstractSearchIndex; k::Int=1) where T
    π = [Vector{Int}() for _ in 1:length(refs.db)]
    # partition((i, p) -> push!(π[p.id], i), dist, objects, refs, k=k)
    partition(objects, refs, k=k) do i, res
        for p in res
            push!(π[p.id], i)
        end
    end
    π
end

"""
    sequence(objects::AbstractVector{T}, refs::AbstractSearchIndex) where T

Computes the nearest reference of each item in the dataset and return it as a sequence of identifiers
"""
function sequence(objects::AbstractVector{T}, refs::AbstractSearchIndex) where T
    s = Vector{Int}(undef, length(objects))
    partition(objects, refs) do i, res
        s[i] = first(res).id
    end
    s
end

