# This file is a part of KCenters.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

using Test

include("loaddata.jl")
using KCenters, SimilaritySearch, MLDataUtils
using StatsBase

@testset "AKNC" begin
    X, ylabels = loadiris()
    le = labelenc(ylabels)
    y = label2ind.(ylabels, le)
    models = Dict()
    best_list = search_params(AKNC, X, y, 16;
        bsize=4,
        mutation_bsize=1,
        ssize=4,
        folds=3,
        search_maxiters=8,
        score=:accuracy,
        tol=-1.0,
        models=models,
        verbose=true,
        ncenters=[3,7],
        k=[1],
        dist=[l2_distance, lp_distance(4)],
        kernel=[direct_kernel],
        minimum_elements_per_centroid=[1, 2]
    )
    @info "========== BEST MODEL =========="
    config, score = best_list[1]
    @test score > 0.9
    @info config, score
    @info get.(models[config], :model, nothing)
end
