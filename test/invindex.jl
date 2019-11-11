using Test

include("loaddata.jl")

@testset "encode by farthest points" begin
    using KCenters, SimilaritySearch
    using StatsBase: mean
    dim = 16
    X = [randn(dim) for i in 1:10000]
    Q = [randn(dim) for i in 1:100]
    dist = l2_distance
    ksearch = 10
    expansion = 3
    P = Performance(dist, X, Q, expected_k=ksearch)
    numcenters = 200
    initial = :fft

    centers = kcenters(dist, X, numcenters, initial=initial)
    index = fit(InvIndex, X, centers, expansion)
    p = probe(P, index, dist)
    @info "before optimization" (recall=p.recall, speedup=p.exhaustive_search_seconds / p.seconds, eval_ratio=p.evaluations / length(X))
    optimize!(index, dist, 0.9, verbose=true)
    p = probe(P, index, dist)
    @info "after optimization" (recall=p.recall, speedup=p.exhaustive_search_seconds / p.seconds, eval_ratio=p.evaluations / length(X))
    # we can expect a small recall reduction since we are using a db's subset for optimizin the index
    @test p.recall > 0.8  
end