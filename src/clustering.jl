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

using SimilaritySearch
export enet, dnet

"""
    enet(dist::Function, X::AbstractVector{T}, numcenters::Int, knr::Int=1) where T

Selects `numcenters` far from each other based on Farthest First Traversal.

- `dist` distance function
- `X` the objects to be computed
- `numcenters` number of centers to be computed
- `knr` number of nearest references per object (knr=1 defines a partition)

Returns a named tuple ``(nn, irefs, dmax)``.

- `irefs` contains the list of centers (indexes to ``X``)
- `seq` contains the ``k`` nearest references for each object in ``X`` (in ``X`` order) 
- `dmax` smallest distance among centers

"""
function enet(dist::Function, X::AbstractVector{T}, numcenters::Int, knr::Int=1) where T
    # refs = Vector{Float64}[]
    irefs = Int[]
    nn = [KnnResult(knr) for i in 1:length(X)]
    dmax = 0.0

    function callback(c, _dmax)
        push!(irefs, c)
        dmax = _dmax
    end

    function capturenn(i, refID, d)
        push!(nn[i], refID, d)
    end

    fftraversal(callback, dist, X, size_criterion(numcenters), capturenn)
    return (irefs=irefs, seq=nn, dmax=dmax)
end


"""
    centroid!(objects::AbstractVector{Vector{F}})::Vector{F} where {F<:Number}

Computes the centroid of the list of objects; use the dot operator (broadcast) to convert several groups of objects
"""
function centroid!(objects::AbstractVector{Vector{F}})::Vector{F} where {F<:Number}
    u = copy(objects[1])
    @inbounds for i in 2:length(objects)
        w = objects[i]
        @simd for j in 1:length(u)
            u[j] += w[j]
        end
    end

    f = 1.0 / length(objects)
    @inbounds @simd for j in 1:length(u)
        u[j] *= f
    end

    return u
end


"""
    dnet(dist::Function, X::AbstractVector{T}, numcenters::Int, knr::Int) where T

Selects `numcenters` far from each other based on density nets.

- `dist` distance function
- `X` the objects to be computed
- `numcenters` number of centers to be computed

Returns a named tuple ``(nn, irefs, dmax)``.

- `irefs` contains the list of centers (indexes to ``X``)
- `dmax` a list of coverage-radius of each center (aligned with irefs centers) smallest distance among centers
- `clusters` the set of items being in the ball defined by the center (aligned with `irefs`)

"""
function dnet(dist::Function, X::AbstractVector{T}, numcenters::Int) where T
    # criterion = change_criterion(0.01)
    n = length(X)
    irefs = Int[]
    dmax = Float64[]
    nn = KnnResult[]

    function callback(c, res)
        push!(irefs, X[c])
        push!(dmax, last(res).dist)
        push!(nn, KnnResult(res))
    end
    
    dnet(callback, dist, X, ceil(Int, n / numcenters))
    (nn, irefs=irefs, dmax=dmax)
end