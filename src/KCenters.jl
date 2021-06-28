# This file is a part of KCenters.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

module KCenters
using SimilaritySearch

include("criterions.jl")
include("centerselection.jl")
include("enet.jl")
include("dnet.jl")
include("utils.jl")
include("clustering.jl")

import Distances: evaluate

export transform, softmax!
"""
    transform(centers::AbstractVector{T}, dmax::AbstractVector, kernel::Function, q::T) where T
    transform(centers::AbstractVector{T}, dmax::AbstractVector, kernel::Function, queries::AbstractVector{T}, normalize!::Function=identity) where T

Creates a vector representation of `q` being its projection along the centrois in `vor` using
the `kernel` function to perform the projection at each center. If a collection of `T`s is given, then it computes `transform`
for each one.
"""
function transform(centers::AbstractVector{T}, dmax::AbstractVector, kernel::Function, q::T) where T
    m = length(centers)
    x = Vector{Float64}(undef, m)
    @inbounds for i in 1:m
        x[i] = kernel(q, centers[i], dmax[i])
    end

    x
end

function transform(
        centers::AbstractVector{T},
        dmax::AbstractVector,
        kernel::Function,
        queries::AbstractVector{T},
        normalize!::Function=identity) where T
    [normalize!(transform(centers, dmax, kernel, q)) for q in queries]
end

"""
    softmax!(vec::AbstractVector)

Inline computation of the softmax function on the input vector
"""
function softmax!(vec::AbstractVector)
    den = 0.0
    @inbounds @simd for v in vec
        den += exp(v)
    end

    den = 1.0 / den
    @inbounds @simd for i in eachindex(vec)
        vec[i] = exp(vec[i]) * den
    end

    vec
end

end