using KCenters
using Documenter

makedocs(;
    modules=[KCenters],
    authors="Eric S. Tellez",
    repo="https://github.com/sadit/KCenters.jl/blob/{commit}{path}#L{line}",
    sitename="KCenters.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", nothing) == "true",
        canonical="https://sadit.github.io/KCenters.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Center selection" => "centerselection.md",
        "Clustering" => "clustering.md",
        "Density nets" => "dnet.md",
        "Epsilon nets" => "enet.md",
        "Stop criterions" => "criterions.md",
        "Partitioning function" => "utils.md"
    ],
    warnonly=true
)

deploydocs(;
    repo="github.com/sadit/KCenters.jl",
    devbranch=nothing,
    branch = "gh-pages",
    versions = ["stable" => "v^", "v#.#.#", "dev" => "dev"]
)
