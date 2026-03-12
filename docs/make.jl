using CasualRCWA
using Documenter

makedocs(;
    sitename = "CasualRCWA",
    pages = Any[
        "Introduction" => "index.md",
        "Docstrings" => "docstrings.md"
    ]
)

deploydocs(
    repo = "github.com/Jeroen-van-der-Meer/CasualRCWA.jl.git",
)
