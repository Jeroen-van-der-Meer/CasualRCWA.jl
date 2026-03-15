using CasualRCWA
using Documenter

makedocs(;
    sitename = "CasualRCWA",
    format = Documenter.HTML(
        assets = ["assets/custom.css"]
    ),
    pages = Any[
        "Introduction" => "index.md",
        "Docstrings" => "docstrings.md"
    ]
)

deploydocs(
    repo = "github.com/Jeroen-van-der-Meer/CasualRCWA.jl.git",
)
