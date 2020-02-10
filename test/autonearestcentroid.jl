# This file is a part of KCenters.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

using Test

using KCenters, SimilaritySearch, MLDataUtils
using StatsBase, JSON

function generate_data()
    X = [rand(2) for i in 1:1000]
    y = [round(Int, 1 + 2 * prod(x)) for x in X]
    X, y
end

@testset "AKNC" begin
    X, y = generate_data()
    X_, y_ = generate_data()

    #models = Dict()

    best_list = search_params(AKNC, X, y, 16;
        bsize=12,
        mutation_bsize=4,
        ssize=4,
        folds=3,
        tol=-1.0,
        search_maxiters=8,
        score=:accuracy,
        #models=models,
        verbose=true,
        ncenters=[0, 3, 7],
        k=[1],
        dist=[l2_distance, l1_distance],
        kernel=[direct_kernel],
        initial_clusters=[:fft, :rand, :dnet],
        minimum_elements_per_centroid=[1, 2]
    )
    @info "========== BEST MODEL =========="
    config, score = best_list[1]
    @test score > 0.9

    # @info get.(models[config], :model, nothing)

    A = fit(config, X, y)
    sa = scores(y_, predict(A, X_))
    B = bagging(config, X, y, ratio=0.5, b=30)
    @test sa.accuracy > 0.85

    @info "class distribution: ", countmap(y), countmap(y_)
    @info config, score
    @info "===== scores for single classifier: $(JSON.json(sa))"

    for k in [1, 5, 7, 9, 11]
        sb = scores(y_, predict(B, X_, k))
        @test sb.accuracy > 0.85
        @info "===== scores for $k: $(JSON.json(sb))"
    end
end

exit(0)