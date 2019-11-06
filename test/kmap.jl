# Copyright 2017-2019 Eric S. Tellez
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

using Test
using SimilaritySearch, KCenters, StatsBase
using Test

include("loaddata.jl")

@testset "Clustering with enet" begin
    X, y = loadiris()
    refs = Vector{Int}()
    dmax = 0.0

    for i in 2:5
        p = enet(l2_distance, X, i^2)
        @info inertia([first(p).dist for p in p.seq])
    end
end

@testset "Clustering with dnet" begin
    X, y = loadiris()
    dmax = 0.0

    for i in 2:5
        res = dnet(l2_distance, X, i^2)
        @info inertia([first(p).dist for p in res.seq])
    end
end
