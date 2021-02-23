var documenterSearchIndex = {"docs":
[{"location":"utils/","page":"Partitioning function","title":"Partitioning function","text":"CurrentModule = KCenters\nDocTestSetup = quote\n    using KCenters\nend","category":"page"},{"location":"utils/#Miscelaneous-functions","page":"Partitioning function","title":"Miscelaneous functions","text":"","category":"section"},{"location":"utils/","page":"Partitioning function","title":"Partitioning function","text":"Helping functions for partitioning data.","category":"page"},{"location":"utils/","page":"Partitioning function","title":"Partitioning function","text":"knr\nsequence\ninvindex\npartition","category":"page"},{"location":"utils/#KCenters.knr","page":"Partitioning function","title":"KCenters.knr","text":"knr(objects::AbstractVector{T}, refs::AbstractSearchContext) where T\n\nComputes an array of k-nearest neighbors for objects\n\n\n\n\n\n","category":"function"},{"location":"utils/#KCenters.sequence","page":"Partitioning function","title":"KCenters.sequence","text":"sequence(objects::AbstractVector{T}, refs::AbstractSearchContext) where T\n\nComputes the nearest reference of each item in the dataset and return it as a sequence of identifiers\n\n\n\n\n\n","category":"function"},{"location":"utils/#KCenters.invindex","page":"Partitioning function","title":"KCenters.invindex","text":"invindex(objects::AbstractVector{T}, refs::AbstractSearchContext; k::Int=1) where T\n\nCreates an inverted index from references to objects. So, an object u is in r's posting list iff r is among the k nearest references of u.\n\n\n\n\n\n","category":"function"},{"location":"utils/#KCenters.partition","page":"Partitioning function","title":"KCenters.partition","text":"partition(callback::Function, objects::AbstractVector{T}, refs::AbstractSearchContext; k::Int=1) where T\n\nGroups items in objects using a nearest neighbor rule over refs. The output is controlled using a callback function. The call is performed in objects order.\n\ncallback is a function that is called for each (objID, refItem)\nobjects is the input dataset\ndist a distance function (T T) rightarrow mathbbR\nrefs the list of references\nk specifies the number of nearest neighbors to use\nindexclass specifies the kind of index to be used, a function receiving (refs, dist) as arguments,   and returning the new metric index\n\nPlease note that each object can be related to more than one group k  1 (default k=1).\n\n\n\n\n\n","category":"function"},{"location":"criterions/","page":"Stop criterions","title":"Stop criterions","text":"CurrentModule = KCenters\nDocTestSetup = quote\n    using KCenters\nend","category":"page"},{"location":"criterions/#Stop-criterions-for-\\epsilon-net-[enet](@ref)","page":"Stop criterions","title":"Stop criterions for epsilon-net enet","text":"","category":"section"},{"location":"criterions/","page":"Stop criterions","title":"Stop criterions","text":"The following functions create an stop criterion function of the form: (dmaxlist, database) -> bool. This function returns true whenever the caller must stop and false otherwise.","category":"page"},{"location":"criterions/","page":"Stop criterions","title":"Stop criterions","text":"size_criterion\nsqrt_criterion\nchange_criterion\nfun_criterion\nlog2_criterion\nepsilon_criterion\nsalesman_criterion","category":"page"},{"location":"criterions/#KCenters.size_criterion","page":"Stop criterions","title":"KCenters.size_criterion","text":"size_criterion(maxsize)\n\nCreates a function that stops when the number of far items are equal or larger than the given maxsize\n\n\n\n\n\n","category":"function"},{"location":"criterions/#KCenters.sqrt_criterion","page":"Stop criterions","title":"KCenters.sqrt_criterion","text":"sqrt_criterion()\n\nStops after sqrt(n) centers\n\n\n\n\n\n","category":"function"},{"location":"criterions/#KCenters.change_criterion","page":"Stop criterions","title":"KCenters.change_criterion","text":"change_criterion(tol=0.001, window=3)\n\nCreates a fuction that stops the process whenever the maximum distance converges (averaging window far items). The tol parameter defines the tolerance range.\n\n\n\n\n\n","category":"function"},{"location":"criterions/#KCenters.fun_criterion","page":"Stop criterions","title":"KCenters.fun_criterion","text":"fun_criterion(fun::Function)\n\nCreates a stop-criterion function that stops whenever the number of far items reaches lceil fun(database)rceil. Already defined examples:\n\n    sqrt_criterion() = fun_criterion(sqrt)\n    log2_criterion() = fun_criterion(log2)\n\n\n\n\n\n","category":"function"},{"location":"criterions/#KCenters.log2_criterion","page":"Stop criterions","title":"KCenters.log2_criterion","text":"log2_criterion()\n\nStops after log_2(n) centers\n\n\n\n\n\n","category":"function"},{"location":"criterions/#KCenters.epsilon_criterion","page":"Stop criterions","title":"KCenters.epsilon_criterion","text":"epsilon_criterion(e)\n\nCreates a function that evaluates the stop criterion when the distance between far items achieves the given e\n\n\n\n\n\n","category":"function"},{"location":"criterions/#KCenters.salesman_criterion","page":"Stop criterions","title":"KCenters.salesman_criterion","text":"salesman_criterion()\n\nIt creates a function that explores the entire dataset making a full farthest first traversal approximation\n\n\n\n\n\n","category":"function"},{"location":"centerselection/","page":"Center selection","title":"Center selection","text":"CurrentModule = KCenters\nDocTestSetup = quote\n    using KCenters\nend","category":"page"},{"location":"centerselection/#Center-selection-schemes","page":"Center selection","title":"Center selection schemes","text":"","category":"section"},{"location":"centerselection/","page":"Center selection","title":"Center selection","text":"KCenters support different ways to compute or select object prototypes (centers). You can use them through center function; any new scheme must specialize from AbstractCenterSelection and specialize center function.","category":"page"},{"location":"centerselection/","page":"Center selection","title":"Center selection","text":"AbstractCenterSelection\nCentroidSelection\nRandomCenterSelection\nMedoidSelection\nKnnCentroidSelection\ncenter","category":"page"},{"location":"centerselection/#KCenters.AbstractCenterSelection","page":"Center selection","title":"KCenters.AbstractCenterSelection","text":"abstract type AbstractCenterSelection end\n\nAbstract type for all center selection strategies\n\n\n\n\n\n","category":"type"},{"location":"centerselection/#KCenters.CentroidSelection","page":"Center selection","title":"KCenters.CentroidSelection","text":"CentroidSelection()\ncenter(sel::CentroidSelection, lst::AbstractVector)\n\nComputes a center as the centroid of the lst set of points\n\n\n\n\n\n","category":"type"},{"location":"centerselection/#KCenters.KnnCentroidSelection","page":"Center selection","title":"KCenters.KnnCentroidSelection","text":"KnnCentroidSelection(sel1::AbstractCenterSelection, sel2::AbstractCenterSelection, dist::PreMetric, k::Int32)\nKnnCentroidSelection(; sel1=CentroidSelection(), sel2=CentroidSelection(), dist=SqL2Distance(), k=0) = KnnCentroidSelection(sel, dist, convert(Int32, k))\ncenter(sel::KnnCentroidSelection, lst::AbstractVector)\n\nComputes a center using the sel1 selection strategy, and computes the final center over the set of k nearest neighbors of the initial center (from lst) using the dist distance function.\n\n\n\n\n\n","category":"type"},{"location":"clustering/","page":"Clustering","title":"Clustering","text":"CurrentModule = KCenters\nDocTestSetup = quote\n    using KCenters\nend","category":"page"},{"location":"clustering/#Clustering","page":"Clustering","title":"Clustering","text":"","category":"section"},{"location":"clustering/","page":"Clustering","title":"Clustering","text":"Functions to cluster objects using distance functions and center selection schemes.","category":"page"},{"location":"clustering/","page":"Clustering","title":"Clustering","text":"ClusteringData\nkcenters\nassociate_centroids","category":"page"},{"location":"clustering/#KCenters.ClusteringData","page":"Clustering","title":"KCenters.ClusteringData","text":"struct ClusteringData{DataType<:AbstractVector}\n    # n elements in the dataset, m centers\n    centers::DataType # centers, m entries\n    freqs::Vector{Int32} # number of elements associated to each center, m entries\n    dmax::Vector{Float32} # stores the distant element associated to each center, m entries\n    codes::Vector{Int32} # id of the associated center, n entries\n    distances::Vector{Float32} # from each element to its nearest center (label), n entries\n    err::Vector{Float32} # dynamic of the error function, at least one entry\nend\n\nThe datastructure output of our clustering procedures\n\n\n\n\n\n","category":"type"},{"location":"clustering/#KCenters.kcenters","page":"Clustering","title":"KCenters.kcenters","text":"kcenters(dist::PreMetric, X::AbstractVector{T}, y::CategoricalArray, sel::AbstractCenterSelection=CentroidSelection()) where T\n\nComputes a center per region (each region is defined by the set of items having the same label in y). The output is compatible with kcenters function when eltype(y) is Int\n\n\n\n\n\nkcenters(dist::PreMetric, X::AbstractVector{T}, k::Integer; sel::AbstractCenterSelection=CentroidSelection(), initial=:fft, maxiters=0, tol=0.001, recall=1.0) where T\nkcenters(dist::PreMetric, X::AbstractVector{T}, C::AbstractzVector{T}; sel::AbstractCenterSelection=CentroidSelection(), maxiters=30, tol=0.001, recall=1.0) where T\n\nPerforms a kcenters clustering of X using dist as distance function and sel to compute center objects. It is based on the Lloyd's algorithm yet using different algorithms as initial clusters.     - :fft the farthest first traversal selects a set of farthest points among them to serve as cluster seeds.     - :dnet the density net algorithm selects a set of points following the same distribution of the datasets; in contrast with a random selection, :dnet ensures that the selected points are not lfloor nk rfloor nearest neighbors.     - :sfft the :fft over a k + log n random sample     - :sdnet the :dnet over a k + log n random sample     - :rand selects the set of random points along the dataset.\n\nIf recall is 1.0 then an exhaustive search is made to find associations of each item to its nearest cluster; if 0  recall  0 then an approximate index (SearchGraph from SimilaritySearch.jl) will be used for the same purpose; the recall controls the expected search quality (trade with search time).\n\n\n\n\n\n","category":"function"},{"location":"clustering/#KCenters.associate_centroids","page":"Clustering","title":"KCenters.associate_centroids","text":"associate_centroids(dist::PreMetric, X, centers)\n\nReturns the named tuple (codes=codes, freqs=freqs, distances=distances, err=s) where codes contains the nearest center index for each item in X under the context of the dist distance function. C is the set of centroids and X the dataset of objects. C also can be provided as a SimilaritySearch's Index.\n\n\n\n\n\n","category":"function"},{"location":"dnet/","page":"Density nets","title":"Density nets","text":"CurrentModule = KCenters\nDocTestSetup = quote\n    using KCenters\nend","category":"page"},{"location":"dnet/#dnet","page":"Density nets","title":"dnet","text":"","category":"section"},{"location":"dnet/","page":"Density nets","title":"Density nets","text":"Functions to compute density nets","category":"page"},{"location":"dnet/","page":"Density nets","title":"Density nets","text":"dnet","category":"page"},{"location":"dnet/#KCenters.dnet","page":"Density nets","title":"KCenters.dnet","text":"dnet(callback::Function, dist::PreMetric, X::AbstractVector{T}, k::Integer) where {T}\n\nA k-net is a set of points M such that each object in X can be:\n\nIt is in M\nIt is in the knn set of an object in M (defined with the distance function dist)\n\nThe size of M is determined by leftceil X  k rightceil\n\nThe dnet function uses the callback function as an output mechanism. This function is called on each center as callback(centerId, res) where res is a KnnResult object (from SimilaritySearch.jl).\n\n\n\n\n\ndnet(dist::PreMetric, X::AbstractVector{T}, numcenters::Integer) where T\n\nSelects numcenters far from each other based on density nets.\n\ndist distance function\nX the objects to be computed\nnumcenters number of centers to be computed\n\nReturns a named tuple (nn irefs dmax).\n\nirefs contains the list of centers (indexes to X)\nseq contains the k nearest references for each object in X (in X order) \ndmax a list of coverage-radius of each center (aligned with irefs centers) smallest distance among centers\n\n\n\n\n\n","category":"function"},{"location":"enet/","page":"Epsilon nets","title":"Epsilon nets","text":"CurrentModule = KCenters\nDocTestSetup = quote\n    using KCenters\nend","category":"page"},{"location":"enet/#enet","page":"Epsilon nets","title":"enet","text":"","category":"section"},{"location":"enet/","page":"Epsilon nets","title":"Epsilon nets","text":"Functions to compute epsilon-nets","category":"page"},{"location":"enet/","page":"Epsilon nets","title":"Epsilon nets","text":"enet","category":"page"},{"location":"enet/#KCenters.enet","page":"Epsilon nets","title":"KCenters.enet","text":"enet(dist::PreMetric, X::AbstractVector{T}, numcenters::Int, knr::Int=1; verbose=false) where T\n\nSelects numcenters far from each other based on Farthest First Traversal.\n\ndist distance function\nX the objects to be computed\nnumcenters number of centers to be computed\nknr number of nearest references per object (knr=1 defines a partition)\n\nReturns a named tuple (nn irefs dmax).\n\nirefs contains the list of centers (indexes to X)\nseq contains the k nearest references for each object in X (in X order) \ndmax smallest distance among centers\n\n\n\n\n\n","category":"function"},{"location":"","page":"Home","title":"Home","text":"CurrentModule = KCenters","category":"page"},{"location":"#KCenters.jl","page":"Home","title":"KCenters.jl","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"The KCenters package implements algorithms that approximate the kcenters problem, in particular, we use farthest first traversal and density nets. It adds several variants of center selection and stopping criterions, along with a number of heuristics and utilities for taking advantage of the resulting groups.","category":"page"},{"location":"","page":"Home","title":"Home","text":"The majority of tasks will use kcenters that outputs a ClusteringData.\nLow level tasks may require to use enet or dnet.\nPartitioning data differently to ClusteringData.\nFor more detailed usage, please visit each one of the pages on the left panel.","category":"page"}]
}
