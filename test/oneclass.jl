using Test

include("loaddata.jl")

@testset "encode by farthest points" begin
    using KCenters, SimilaritySearch
    using StatsBase: mean

    X, ylabels = loadiris()
    dist = l2_distance #lp_distance(3.3)
    L = Float64[]
    for label in ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
        y = ylabels .== label
        A = X[y]
        B = X[.~y]
        k = 5
        centers = kcenters(dist, A, k, verbose=true, maxiters=3) # TODO Add more control to kcenters or use a kcenters output as input
        occ = fit(OneClassClassifier, centers)
        ypred = [predict(occ, dist, x).similarity > 0 for x in X]
        push!(L, mean(ypred .== y))
        println(stderr, "==> $label: $(L[end])")
    end

    macrorecall = mean(L)
    println(stderr, "===> macro-recall: $macrorecall")
    @test macrorecall > 0.9
end