# shamlessly ~~stolen from~~ inspired by Krotov.jl

using SparseExtra, Documenter
import Pkg
DocMeta.setdocmeta!(SparseExtra, :DocTestSetup, 
    :(using SparseExtra); recursive=true)
PROJECT_TOML = Pkg.TOML.parsefile(joinpath(@__DIR__, "..", "Project.toml"))
VERSION = PROJECT_TOML["version"]
NAME = PROJECT_TOML["name"]
AUTHORS = join(PROJECT_TOML["authors"], ", ") * " and contributors"
GITHUB = "https://github.com/SobhanMP/SparseExtra.jl"

println("Starting makedocs")

makedocs(;
    authors=AUTHORS,
    sitename="SparseExtra.jl",
    modules=[SparseExtra],
    format=Documenter.HTML(;
        prettyurls=true,
        canonical="https://SobhanMP.github.io/SparseExtra.jl",
        assets=String[],
        footer="[$NAME.jl]($GITHUB) v$VERSION docs powered by [Documenter.jl](https://github.com/JuliaDocs/Documenter.jl).",
        mathengine=KaTeX()
    ),
    pages=[
        "Home" => "index.md",
        "`iternz`" => "iternz.md"
    ]
)
