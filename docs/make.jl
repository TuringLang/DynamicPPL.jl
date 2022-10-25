using Documenter
using DynamicPPL
using DynamicPPL: AbstractPPL

# Doctest setup
DocMeta.setdocmeta!(
    DynamicPPL, :DocTestSetup, :(using DynamicPPL, Distributions); recursive=true
)

makedocs(;
    sitename="DynamicPPL",
    format=Documenter.HTML(),
    modules=[DynamicPPL],
    pages=["Home" => "index.md", "API" => "api.md"],
    strict=true,
    checkdocs=:exports,
)

deploydocs(; repo="github.com/TuringLang/DynamicPPL.jl.git", push_preview=true)
