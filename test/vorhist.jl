using Test

include("loaddata.jl")
using KCenters, SimilaritySearch
using StatsBase: mean

@testset "One class classifier with DeloneHistogram" begin

    X, ylabels = loadiris()
    dist = l2_distance

    for k in [3, 5, 7, 11]
        println("===> k=$k")
        for initial in [:fft, :dnet, :rand], maxiters in [0, 3, 10]
            L = Float64[]
            for label in ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
                y = ylabels .== label
                A = X[y]
                centers = kcenters(dist, A, k, initial=initial, maxiters=maxiters, tol=0.0)
                vor = fit(DeloneHistogram, centers)
                ypred = predict(vor, dist, X)
                push!(L, mean(ypred .== y))
                @test L[end] > 0.6
            end

            macrorecall = mean(L)
            println(stderr, "===> (k=$k, initial=$initial, maxiters=$maxiters); macro-recall: $macrorecall")
            @test macrorecall > 0.8
        end
    end
end

@testset "NearestCentroid with DeloneHistogram" begin
    X, ylabels = loadiris()
    M = Dict(label => i for (i, label) in enumerate(unique(ylabels) |> sort!))
    y = [M[y] for y in ylabels]
    dist = l2_distance
    for kernel in [gaussian_kernel, laplacian_kernel, cauchy_kernel, sigmoid_kernel, tanh_kernel, relu_kernel]
        C = kcenters_by_label(dist, X, y)
        @info "XXXXXX>", (kernel, dist)

        D = fit(DeloneHistogram, C)

        # @show transform(D.centers.db, D.dmax, kernel(dist), X[1:3], softmax!) ## TODO add some test, right now we only ensure that it runs
        nc = fit(NearestCentroid, D)
        ypred = predict(nc, kernel(dist), X)
        
        @info mean(ypred .== y)
    end
end

exit(0)