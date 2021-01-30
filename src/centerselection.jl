# This file is a part of KCenters.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

using Random

abstract type AbstractCenterSelection end
export AbstractCenterSelection, CentroidSelection, RandomCenterSelection, MedoidSelection, KnnCentroidSelection

struct CentroidSelection <: AbstractCenterSelection end
struct RandomCenterSelection <: AbstractCenterSelection end

struct MedoidSelection{M_<:PreMetric} <: AbstractCenterSelection
    dist::M_
    ratio::Float32
end

struct KnnCentroidSelection{S_<:AbstractCenterSelection, M_<:PreMetric} <: AbstractCenterSelection
    sel::S_
    dist::M_
    k::Int32
end

StructTypes.StructType(::Type{<:CentroidSelection}) = StructTypes.Struct()
StructTypes.StructType(::Type{<:RandomCenterSelection}) = StructTypes.Struct()
StructTypes.StructType(::Type{<:MedoidSelection}) = StructTypes.Struct()
StructTypes.StructType(::Type{<:KnnCentroidSelection}) = StructTypes.Struct()

MedoidSelection(; dist=SqL2Distance(), ratio=0.5) = MedoidSelection(dist, convert(Float32, ratio))
KnnCentroidSelection(; sel=CentroidSelection(), dist=SqL2Distance(), k=0) = KnnCentroidSelection(sel, dist, convert(Int32, k))

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
    c = center(sel.sel, lst)
    seq = ExhaustiveSearch(sel.dist, lst)
    k = sel.k == 0 ? ceil(Int32, log2(length(lst))) : sel.k
    mean(lst[[p.id for p in search(seq, c, k)]])
end
