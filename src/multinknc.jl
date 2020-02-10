# This file is a part of KCenters.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

using Random
export glue, bagging

"""
    glue(arr::AbstractVector{K}) where K <: KNC
    glue(arr::AbstractVector{K}) where K <: AKNC

In its first form it joins a list of KNC classifiers into a single one.
The second form, it also joins the classifiers into a single AKNC classifier;
however, it uses takes specifications (the kernel function and the configuration)
of the first element of the list.

"""
function glue(arr::AbstractVector{K}) where K <: KNC
    centers = vcat([c.centers for c in arr]...)
    dmax = vcat([c.dmax for c in arr]...)
    class_map = vcat([c.class_map for c in arr]...)
    KNC(centers, dmax, class_map, arr[1].nclasses)
end

function glue(arr::AbstractVector{K}) where K <: AKNC
    a = first(arr)
    nc = glue([s.nc for s in arr])
    AKNC(nc, a.kernel, a.config)
end

"""
    bagging(config::AKNC_Config, X::AbstractVector, y::AbstractVector{I}; b=13, ratio=0.5) where {I<:Integer}

Creates `b` classifiers, each trained with a random `ratio` of the dataset;
these classifiers are joint into a single classifier with `glue`.
"""
function bagging(config::AKNC_Config, X::AbstractVector, y::AbstractVector{I}; b=13, ratio=0.5) where {I<:Integer}
    indexes = collect(1:length(X))
    m = ceil(Int, ratio * length(X))

    L = Vector{AKNC}(undef, b)
    for i in 1:b
        shuffle!(indexes)
        sample = @view indexes[1:m]
        L[i] = fit(AKNC, config, X[sample], y[sample])
    end

    glue(L)
end
