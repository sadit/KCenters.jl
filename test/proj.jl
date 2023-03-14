# This file is a part of KCenters.jl

using Test, JET
using Random, SimilaritySearch, KCenters, StatsBase


@testset "Projections" begin
    db = MatrixDatabase(rand(Float32, 4, 10000))
    queries = MatrixDatabase(rand(Float32, 4, 100))
    dist = L2Distance()
    k = 15

    Igold, _ = searchbatch(ExhaustiveSearch(; dist, db), queries, k)
    
    refs = let
        R = kcenters(dist, db, 64; initial=:rand)
        refs = MatrixDatabase(R.centers)
        ExhaustiveSearch(; dist, db=refs)
    end

  let
    @info "KNR"
    knrsearch = 8
    knr = Knr(Int8, refs; sort=true, k=knrsearch)
    A = encode_database(knr, db)
    B = encode_database(knr, queries)
    @test size(A) == (knrsearch, length(db))
    display(A[:, 1:3])
    E = ExhaustiveSearch(; dist=JaccardDistance(), db=MatrixDatabase(A))
    I, _ = searchbatch(E, MatrixDatabase(B), k)
    recall = macrorecall(Igold, I)
    @show recall
    @test recall > 0.2
  end
  
  let
    @info "Perms"
    P = Perms(SqL2Distance(), database(refs))
    A = encode_database(P, db)
    B = encode_database(P, queries)
    @show size(A), size(B)
    E = ExhaustiveSearch(; dist=SqL2Distance(), db=MatrixDatabase(A))
    I, _ = searchbatch(E, MatrixDatabase(B), k)
    recall = macrorecall(Igold, I)
    @show recall
    @test recall > 0.2
  end
  
  let
    @info "BinPerms"
    P = BinPerms(SqL2Distance(), database(refs))
    A = encode_database(P, db)
    B = encode_database(P, queries)
    @show size(A), size(B)
    E = ExhaustiveSearch(; dist=BinaryHammingDistance(), db=MatrixDatabase(A))
    I, _ = searchbatch(E, MatrixDatabase(B), k)
    recall = macrorecall(Igold, I)
    @show recall
    @test recall > 0.5
  end
  
  let
    @info "BinWalk"
    P = BinWalk(SqL2Distance(), database(refs))
    A = encode_database(P, db)
    B = encode_database(P, queries)
    @show size(A), size(B)
    E = ExhaustiveSearch(; dist=BinaryHammingDistance(), db=MatrixDatabase(A))
    I, _ = searchbatch(E, MatrixDatabase(B), k)
    recall = macrorecall(Igold, I)
    @show recall
    @test recall > 0.5
  end
end
