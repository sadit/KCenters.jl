using Test

include("loaddata.jl")
using KCenters, SimilaritySearch
using StatsBase

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

@testset "NearestCentroid" begin
    X, ylabels = loadiris()
    M = Dict(label => i for (i, label) in enumerate(unique(ylabels) |> sort!))
    y = [M[y] for y in ylabels]
    dist = lp_distance(0.7)
    for kernel in [gaussian_kernel, laplacian_kernel, cauchy_kernel, sigmoid_kernel, tanh_kernel, relu_kernel]
        C = kcenters(dist, X, y)
        @info "XXXXXX>", (kernel, dist)

        D = fit(DeloneHistogram, C)
        nc = fit(NearestCentroid, D)
        ypred = predict(nc, kernel(dist), X)
        @test mean(ypred .== y) > 0.8
    end

    for kernel in [gaussian_kernel, laplacian_kernel, cauchy_kernel, sigmoid_kernel, tanh_kernel, relu_kernel]
        @info "XXXXXX====>", (kernel, dist)

        C = kcenters(dist, X, 21)
        D = fit(DeloneInvIndex, X, C, 1)
        nc = fit(NearestCentroid, D, y)
        @show nc.class_map
        ypred = predict(nc, kernel(dist), X)
        @test mean(ypred .== y) > 0.8
    end
end

