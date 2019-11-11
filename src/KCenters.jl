module KCenters

include("criterions.jl")
include("fftraversal.jl")
include("dnet.jl")
include("utils.jl")
include("vorhist.jl")
include("clustering.jl")
include("invindex.jl")
include("kernels.jl")

export transform
"""
    transform(vor::UnionAll{DeloneHistogram,DeloneInvIndex}, kernel::Function, q::T) where T

Creates a vector representation of `q` being its projection along the centrois in `vor` using
the `kernel` function to perform the projection at each center.
"""
function transform(vor::Union{DeloneHistogram,DeloneInvIndex}, kernel::Function, q::T) where T
    C = vor.centers.db
    m = length(C)
    x = Vector{Float64}(undef, m)
    @inbounds for i in 1:m
        d = kernel(q, C[i])
        x[i] = kernel(q, C[i], vor.dmax[i])
    end

    x
end

end
