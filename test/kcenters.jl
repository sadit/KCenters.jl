    # This file is a part of KCenters.jl

using Test
using Random, SimilaritySearch, KCenters, StatsBase

const X = MatrixDatabase(rand(Float32, 4, 10000))

@testset "Clustering with enet" begin
    for i in 2:5
        p = enet(L2Distance(), X, i^2)
        i == 5 && enet(L2Distance(), X, i^2)
        D = minimum.(p.seq)
        @test maximum(D) <= p.dmax
        # minimum.(p.seq)
        #@info p.irefs
        #@info unique(p.irefs)
    end
end

@testset "Clustering with dnet" begin
    for i in 2:5
        res = dnet(L2Distance(), X, i^2)
        #i == 5 && (@test_call dnet(L2Distance(), X, i^2))
        @info mean([minimum(p) for p in res.seq])
    end
end

@testset "Clustering with KCenters" begin
    cfft = kcenters(L2Distance(), X, 16)
    cdnet = kcenters(L2Distance(), X, 16, initial=:dnet)
    crand = kcenters(L2Distance(), X, 16, initial=:rand)
   # @test_call kcenters(L2Distance(), X, 16)
    #@test_call kcenters(L2Distance(), X, 16, initial=:dnet)
    #@test_call kcenters(L2Distance(), X, 16, initial=:rand)
    @show mean(cfft.distances)
    @show mean(cdnet.distances)
    @show mean(crand.distances)
end


@testset "Clustering with KCenters; KnnCentroidSelection" begin
    c1 = kcenters(L2Distance(), X, 16; sel=KnnCentroidSelection())
    c2 = kcenters(L2Distance(), X, 16; sel=KnnCentroidSelection(sel1=MedoidSelection()))
    c3 = kcenters(L2Distance(), X, 16; sel=KnnCentroidSelection(sel1=RandomCenterSelection()))
    d1 = mean(c1.distances)
    d2 = mean(c2.distances)
    d3 = mean(c3.distances)
    @show d1 d2 d3
    @test abs(d1 - d2) < 0.2
    @test abs(d1 - d3) < 0.2
end

@testset "Clustering with KCenters with an approximate index" begin
    cfft = kcenters(L2Distance(), X, 16, recall=0.99)
    cdnet = kcenters(L2Distance(), X, 16, initial=:dnet, recall=0.99)
    crand = kcenters(L2Distance(), X, 16, initial=:rand, recall=0.99)
    d1 = mean(cfft.distances)
    d2 = mean(cdnet.distances)
    d3 = mean(crand.distances)
    @show d1 d2 d3
    @test abs(d1 - d2) < 0.2
    @test abs(d1 - d3) < 0.2
end
