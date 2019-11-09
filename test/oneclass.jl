using Test

include("loaddata.jl")

@testset "encode by farthest points" begin
    using KCenters, SimilaritySearch
    using StatsBase: mean

    X, ylabels = loadiris()
    dist = l2_distance

    for k in [3, 5, 7, 11]
        println("===> k=$k")
        for initial in [:fft, :dnet, :rand], maxiters in [0, 3, 10]
            L = Float64[]
            for label in ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
                y = ylabels .== label
                A = X[y]
                centers = kcenters(dist, A, k, initial=initial, maxiters=maxiters, tol=0.0) # TODO Add more control to kcenters or use a kcenters output as input
                occ = fit(OneClassClassifier, centers)
                ypred = [predict(occ, dist, x).similarity > 0 for x in X]
                @show predict(occ, dist, X[1])
                push!(L, mean(ypred .== y))
                @test L[end] > 0.7
            end

            macrorecall = mean(L)
            println(stderr, "===> (k=$k, initial=$initial, maxiters=$maxiters); macro-recall: $macrorecall")
            @test macrorecall > 0.8
        end
    end
end