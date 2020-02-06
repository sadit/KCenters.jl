# This file is a part of KCenters.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

using Test
using SimilaritySearch, KCenters, StatsBase
using Test

include("loaddata.jl")

X, y = loadiris()

@testset "Clustering with enet" begin
    for i in 2:5
        p = enet(l2_distance, X, i^2)
        D = [first(p).dist for p in p.seq]
        @test maximum(D) <= p.dmax
    end
end

@testset "Clustering with dnet" begin
    for i in 2:5
        res = dnet(l2_distance, X, i^2)
        @info mean([first(p).dist for p in res.seq])
    end
end

@testset "Clustering with KCenters" begin
    cfft = KCenters.kcenters(l2_distance, X, 16)
    cdnet = KCenters.kcenters(l2_distance, X, 16, initial=:dnet)
    crand = KCenters.kcenters(l2_distance, X, 16, initial=:rand)
    @show mean(cfft.distances)
    @show mean(cdnet.distances)
    @show mean(crand.distances)
end


@testset "Clustering with KCenters with an approximate index" begin
    cfft = KCenters.kcenters(l2_distance, X, 16, recall=0.99)
    cdnet = KCenters.kcenters(l2_distance, X, 16, initial=:dnet, recall=0.99)
    crand = KCenters.kcenters(l2_distance, X, 16, initial=:rand, recall=0.99)
    @show mean(cfft.distances)
    @show mean(cdnet.distances)
    @show mean(crand.distances)
end
