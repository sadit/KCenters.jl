using SimilaritySearch
export kmap, partition, knr, sequence, fftclustering, centroid!

"""
    kmap(objects::AbstractVector{T}, kernel, refs::AbstractVector{T}) where {T}

Transforms `objects` to a new representation space induced by ``(refs, dist, kernel)``
- `refs` a list of references
- `kernel` a kernel function (and an embedded distance) with signature ``(T, T) \\rightarrow Float64``
"""
function kmap(objects::AbstractVector{T}, kernel, refs::AbstractVector{T}) where {T}
    # X = Vector{T}(length(objects))
    m = Vector{Vector{Float64}}(undef, length(objects))
    @inbounds for i in 1:length(objects)
        u = Vector{Float64}(undef, length(refs))
        obj = objects[i]
        for j in 1:length(refs)
            u[j] = kernel(obj, refs[j])
        end

        m[i] = u
    end

    return m
end

"""
    partition(callback::Function, dist::Function, objects::AbstractVector{T}, refs::Index; k::Int=1) where T

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
function partition(callback::Function, dist::Function, objects::AbstractVector{T}, refs::Index; k::Int=1) where T
    res = KnnResult(k)
    for i in 1:length(objects)
        empty!(res)
        callback(i, search(refs, dist, objects[i], res))
    end
end

"""
    invindex(dist::Function, objects::AbstractVector{T}, refs::Index; k::Int=1) where T

Creates an inverted index from references to objects.
So, an object ``u`` is in ``r``'s posting list iff ``r``
is among the ``k`` nearest references of ``u``.

"""
function invindex(dist::Function, objects::AbstractVector{T}, refs::Index; k::Int=1) where T
    π = [Vector{Int}() for i in 1:length(refs.db)]
    # partition((i, p) -> push!(π[p.objID], i), dist, objects, refs, k=k)
    partition(dist, objects, refs, k=k) do i, res
        for p in res
            push!(π[p.objID], i)
        end
    end
    π
end

"""
    sequence(dist::Function, objects::AbstractVector{T}, refs::Index) where T

Computes the nearest reference of each item in the dataset and return it as a sequence of identifiers
"""
function sequence(dist::Function, objects::AbstractVector{T}, refs::Index) where T
    s = Vector{Int}(undef, length(objects))
    partition(dist, objects, refs) do i, res
        s[i] = first(res).objID
    end
    s
end

"""
    knr(dist::Function, objects::AbstractVector{T}, refs::Index) where T

Computes an array of k-nearest neighbors for `objects`
"""
function knr(dist::Function, objects::AbstractVector{T}, refs::Index) where T
    s = Vector{Vector{Int}}(undef, length(objects))
    partition(dist, objects, refs) do i, res
        s[i] = [p.objID for p in res]
    end
    s
end
