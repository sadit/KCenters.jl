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
        d = [first(p).dist for p in p.seq]
        @info inertia(d)
        @test maximum(d) <= p.dmax
    end
end

@testset "Clustering with dnet" begin
    for i in 2:5
        res = dnet(l2_distance, X, i^2)
        @info inertia([first(p).dist for p in res.seq])
    end
end


@testset "Clustering with KCenters" begin
    cfft = KCenters.kcenters(l2_distance, X, 16)
    cdnet = KCenters.kcenters(l2_distance, X, 16, initial=:dnet)
    crand = KCenters.kcenters(l2_distance, X, 16, initial=:random)
    @show inertia(cfft.distances)
    @show inertia(cdnet.distances)
    @show inertia(crand.distances)
end


@testset "Clustering with KCenters with an approximate index" begin
    cfft = KCenters.kcenters(l2_distance, X, 16, recall=0.99)
    cdnet = KCenters.kcenters(l2_distance, X, 16, initial=:dnet, recall=0.99)
    crand = KCenters.kcenters(l2_distance, X, 16, initial=:random, recall=0.99)
    @show inertia(cfft.distances)
    @show inertia(cdnet.distances)
    @show inertia(crand.distances)
end
