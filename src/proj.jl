# This file is a part of KCenters.jl
#
# Projections based on references (computed with kcenters)

using Polyester

export AbstractReferenceMapping, Knr, Perms, BinPerms, BinWalk
export encode_database, encode_database!, encode_object, encode_object!, encode_object_res!

abstract type AbstractReferenceMapping end

struct PermsCacheEncoder
    P::Vector{Int32}
    invP::Vector{Int32}
    vec::Vector{Float32}
    
    function PermsCacheEncoder(permsize)
        n = permsize
        new(zeros(Int32, n), zeros(Int32, n), zeros(Float32, n))
    end
end

function invperm!(invP, P)
    for i in 1:length(P)
        invP[P[i]] = i
    end
 
    invP
end

const PERMS_CACHES = [PermsCacheEncoder(16)]
const KNR_CACHES = [KnnResult(16)]

function __init__proj_cache()
  while length(PERMS_CACHES) < Threads.nthreads()
    push!(PERMS_CACHES, PermsCacheEncoder(16))
    push!(KNR_CACHES, KnnResult(16))
  end
end

function getpermscache(m)
  c = PERMS_CACHES[Threads.threadid()]
  resize!(c.P, m)
  resize!(c.invP, m)
  resize!(c.vec, m)
  c
end

function getknrcache(k::Integer, pools=nothing)
    reuse!(KNR_CACHES[Threads.threadid()], k)
end


struct Knr{IndexType<:AbstractSearchIndex,IntType<:Integer} <: AbstractReferenceMapping
  itype::Type{IntType}
  refs::IndexType
  k::Int32
  sort::Bool
end
  
function Knr(::Type{IntType}, refs::AbstractSearchIndex; k::Integer=8, sort=false) where {IntType<:Integer}
    k <= length(refs) || throw(ArgumentError("the dim of the output matrix should be smaller or equal than the number of references"))
    Knr(IntType, refs, convert(Int32, k), convert(Bool, sort))
end

"""
    encode_database!(knr::Knr, out::Matrix, S::AbstractDatabase, refs::AbstractSearchIndex; minbatch) 
    encode_database(knr::Knr, S::AbstractDatabase, refs::AbstractSearchIndex, k; minbatch) 

Computes a `k` nearest refertences projection of `S` using the indexed references `refs` 
"""
function encode_database!(knr::Knr, out::AbstractMatrix, S::AbstractDatabase; minbatch=0)
    k, n = size(out)
    n == length(S) || throw(ArgumentError("output and input sizes must match"))

    minbatch = SimilaritySearch.getminbatch(minbatch, n)
    @batch per=thread minbatch=minbatch for i in 1:n
      encode_object!(knr, view(out, :, i), S[i])
    end

    out
end

function encode_object!(knr::Knr, out::AbstractVector, obj)
  k = length(out)
  res = getknrcache(k)
  search(knr.refs, obj, res)
  k_ = length(res)
  o = @view out[1:k_]
  o .= IdView(res)
  knr.sort && sort!(o)
  k_ < k && (out[k_+1:end] .= zero(eltype(out)))
  out
end

function encode_database(knr::Knr, S::AbstractDatabase; k::Integer=knr.k, minbatch=0)
  X = Matrix{knr.itype}(undef, k, length(S))
  encode_database!(knr, X, S; minbatch)
end

function encode_object(knr::Knr, obj; k::Integer=knr.k)
  X = Vector{knr.itype}(undef, k)
  encode_object!(knr, X, obj)
end

encode_object_res!(knr::Knr, res::KnnResult, obj) = search(knr.refs, obj, res).res
encode_object_res!(knr::Knr, obj; k=knr.k) = search(knr.refs, obj, getknrcache(k)).res

struct Perms{DistType<:SemiMetric,DbType<:AbstractDatabase} <: AbstractReferenceMapping
    dist::DistType
    refs::DbType
    pool::Matrix{Int32}

    function Perms(dist::SemiMetric, refs::AbstractDatabase; nperms::Integer=2, permsize::Integer=16)
        2 <= permsize <= length(refs) || throw(ArgumentError("invalid permsize $permsize"))
        pool = Matrix{Int32}(undef, permsize, nperms)
        perm = Vector{Int32}(1:length(refs))

        for i in 1:nperms
            shuffle!(perm)
            pool[:, i] .= view(perm, 1:permsize)
        end
        
        new{typeof(dist), typeof(refs)}(dist, refs, pool)
    end
end

@inline permsize(M::Perms) = size(M.pool, 1)
@inline nperms(M::Perms) = size(M.pool, 2)

function encode_object!(M::Perms, vout, obj)
    cache = getpermscache(permsize(M))
    for i in 1:nperms(M)
        col = view(M.pool, :, i)
        for (j, k) in enumerate(col)
            cache.vec[j] = evaluate(M.dist, M.refs[k], obj)
        end
        
        sortperm!(cache.P, cache.vec)
        invperm!(cache.invP, cache.P)
        vout[:, i] .= cache.invP
    end
    
    vout
end

function encode_object(M::Perms, obj)
    out = Vector{Float32}(undef, permsize(M), nperms(M))
    encode_object!(M, out, obj)
end

function encode_database!(M::Perms, out::Matrix, S::AbstractDatabase; minbatch=0)
  minbatch = SimilaritySearch.getminbatch(minbatch, length(S))
    @batch per=thread minbatch=minbatch for i in eachindex(S)
        x = reshape(view(out, :, i), permsize(M), nperms(M))
        encode_object!(M, x, S[i])
    end

    out
end

function encode_database(M::Perms, S::AbstractDatabase; minbatch=0)
    D = Matrix{Float32}(undef, permsize(M) * nperms(M), length(S))
    encode_database!(M, D, S; minbatch)
end

struct BinPerms{DistType<:SemiMetric,DbType<:AbstractDatabase} <: AbstractReferenceMapping
    dist::DistType
    refs::DbType
    pool::Matrix{Int32}
    shift::Int
    
    function BinPerms(dist::SemiMetric, refs::AbstractDatabase; nperms::Integer=16, permsize::Integer=64, shift::Integer=permsize ÷ 3)
        2 <= permsize <= 64 || throw(ArgumentError("invalid permsize $permsize"))
        numrefs = length(refs)
        numrefs >= permsize || throw(ArgumentError("invalid numrefs; it should follows numrefs($numrefs) >= permsize($permsize)"))
        pool = Matrix{Int32}(undef, permsize, nperms)
        perm = Vector{Int32}(1:numrefs)

        for i in 1:nperms
            shuffle!(perm)
            pool[:, i] .= view(perm, 1:permsize)
        end
        
        new{typeof(dist), typeof(refs)}(dist, refs, pool, shift)
    end
end

@inline permsize(M::BinPerms) = size(M.pool, 1)
@inline nperms(M::BinPerms) = size(M.pool, 2)
@inline shift(M::BinPerms) = M.shift

function encode_object!(M::BinPerms, vout, obj)
    cache = getpermscache(permsize(M))
    for i in 1:nperms(M)
        col = view(M.pool, :, i)
        for (j, k) in enumerate(col)
            cache.vec[j] = evaluate(M.dist, M.refs[k], obj)
        end
        
        sortperm!(cache.P, cache.vec)
        invperm!(cache.invP, cache.P)
        
        E = zero(UInt64)
        for j in 1:permsize(M)
            s = abs(cache.invP[j] - j) > shift(M)
            E |= s << (j-1)
        end
        
        vout[i] = E
    end
    
    vout
end

function encode_object(M::BinPerms, obj)
    out = Vector{UInt64}(undef, nperms(M))
    encode_object!(M, out, obj)
end

function encode_database!(M::BinPerms, out::Matrix, db::AbstractDatabase; minbatch=0) 
  minbatch = getminbatch(minbatch, length(db))
    @batch per=thread minbatch=minbatch for i in eachindex(db)
        encode_object!(M, view(out, :, i), db[i])
    end

    out
end

function encode_database(M::BinPerms, db::AbstractDatabase; minbatch=0)
    out = Matrix{UInt64}(undef, nperms(M), length(db))
    encode_database!(M, out, db; minbatch)
end


struct BinWalk{DistType<:SemiMetric,DbType<:AbstractDatabase} <: AbstractReferenceMapping
    dist::DistType
    refs::DbType
    pool::Matrix{Int32}
        
    function BinWalk(dist::SemiMetric, refs::AbstractDatabase; nperms::Integer=16, permsize::Integer=64)
        2 <= permsize <= 64 || throw(ArgumentError("invalid permsize $permsize"))
        numrefs = length(refs)
        numrefs >= permsize || throw(ArgumentError("invalid numrefs; it should follows numrefs($numrefs) >= permsize($permsize)"))
        pool = Matrix{Int32}(undef, permsize, nperms)
        perm = Vector{Int32}(1:numrefs)

        for i in 1:nperms
            shuffle!(perm)
            pool[:, i] .= view(perm, 1:permsize)
        end
        
        new{typeof(dist), typeof(refs)}(dist, refs, pool)
    end
end

@inline permsize(M::BinWalk) = size(M.pool, 1)
@inline nperms(M::BinWalk) = size(M.pool, 2)

function encode_object!(M::BinWalk, vout::AbstractVector, obj)
    cache = getpermscache(permsize(M))

    for i in 1:nperms(M)
        col = view(M.pool, :, i)
        for (j, k) in enumerate(col)
            cache.vec[j] = evaluate(M.dist, M.refs[k], obj)
        end
                
        E = zero(UInt64)
        for j in 1:permsize(M)-1
            s = cache.vec[j] < cache.vec[j+1]
            E |= s << (j-1)
        end

        j = permsize(M)
        s = cache.vec[j] < cache.vec[1]  # circular comparison
        E |= s << (j-1)
        
        vout[i] = E
    end
    
    vout
end

function encode_object(M::BinWalk, obj)
    out = Vector{UInt64}(undef, nperms(M))
    encode_object!(M, out, obj)
end

function encode_database!(M::BinWalk, out::Matrix, db::AbstractDatabase; minbatch=0)
  minbatch = getminbatch(minbatch, length(db))
    @batch per=thread minbatch=minbatch for i in eachindex(db)
        encode_object!(M, view(out, :, i), db[i])
    end

    out
end

function encode_database(M::BinWalk, db::AbstractDatabase)
    out = Matrix{UInt64}(undef, nperms(M), length(db))
    encode_database!(M, out, db)
end

