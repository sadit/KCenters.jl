# This file is a part of KCenters.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

using Random

export AbstractCenterSelection,
    CentroidSelection,
    RandomCenterSelection,
    MedoidSelection,
    KnnCentroidSelection,
    center

""" 
    abstract type AbstractCenterSelection end

Abstract type for all center selection strategies
"""
abstract type AbstractCenterSelection end

"""
    CentroidSelection()
    center(sel::CentroidSelection, lst::AbstractVector)

Computes a center as the centroid of the `lst` set of points
"""
struct CentroidSelection <: AbstractCenterSelection end


"""
    RandomCenterSelection()
    center(sel::RandomCenterSelection, lst::AbstractVector)

Selects a random element from `lst` as representing center
"""

struct RandomCenterSelection <: AbstractCenterSelection end

"""
    MedoidSelection(dist::PreMetric, ratio::Float32)
    MedoidSelection(; dist=SqL2Distance(), ratio=0.5) = MedoidSelection(dist, convert(Float32, ratio))
    center(sel::MedoidSelection, lst::AbstractVector)

Computes the medoid of lst; if ``0 < ratio < 1`` then a sampling of `lst` (``ratio * |lst|`` elements) is used instead of the complete set
"""

struct MedoidSelection{M_<:PreMetric} <: AbstractCenterSelection
    dist::M_
    ratio::Float32
end


"""
    KnnCentroidSelection(sel1::AbstractCenterSelection, sel2::AbstractCenterSelection, dist::PreMetric, k::Int32)
    KnnCentroidSelection(; sel1=CentroidSelection(), sel2=CentroidSelection(), dist=SqL2Distance(), k=0) = KnnCentroidSelection(sel, dist, convert(Int32, k))
    center(sel::KnnCentroidSelection, lst::AbstractVector)

Computes a center using the `sel1` selection strategy, and computes the final center over the set of `k` nearest neighbors
of the initial center (from `lst`) using the `dist` distance function.
"""
struct KnnCentroidSelection{S1_<:AbstractCenterSelection, S2_<:AbstractCenterSelection, M_<:PreMetric} <: AbstractCenterSelection
    sel1::S1_
    sel2::S2_
    dist::M_
    k::Int32
end

MedoidSelection(; dist=SqL2Distance(), ratio=0.5) = MedoidSelection(dist, convert(Float32, ratio))
KnnCentroidSelection(; sel1=CentroidSelection(), sel2=CentroidSelection(), dist=SqL2Distance(), k=0) = KnnCentroidSelection(sel1, sel2, dist, convert(Int32, k))

center(::CentroidSelection, lst::AbstractVector{ObjectType}) where {ObjectType<:AbstractVector{N}} where {N<:Real} =
    mean(lst)

center(::RandomCenterSelection, lst::AbstractVector) = rand(lst)

function center(sel::MedoidSelection, lst::AbstractVector)
    if sel.ratio < 1.0
        ss = randsubseq(1:length(lst), sel.ratio)
        if length(ss) > 0
            lst = lst[ss]
        end
    end
    
    L = zeros(Float32, length(lst))
    n = length(lst)
    for i in 1:n
        for j in i+1:length(n)
            d = evaluate(sel.dist, lst[i], lst[j])
            L[i] += d
            L[j] += d
        end
    end

    lst[argmin(L)]
end

function center(sel::KnnCentroidSelection, lst::AbstractVector)
    c = center(sel.sel1, lst)
    seq = ExhaustiveSearch(sel.dist, lst)
    k = sel.k == 0 ? ceil(Int32, log2(length(lst))) : sel.k
    k = max(1, k)
    center(sel.sel2, lst[[id for (id, dist) in search(seq, c, k)]])
end
