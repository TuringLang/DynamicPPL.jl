using Documenter
using DynamicPPL
using DynamicPPL: AbstractPPL
# NOTE: This is necessary to ensure that if we print something from
# Distributions.jl in a doctest, then the shown value will not include
# a qualifier; that is, we don't want `Distributions.Normal{Float64}`
# but rather `Normal{Float64}`. The latter is what will then be printed
# in the doctest as run in `test/runtests.jl`, and so we need to stay
# consistent with that.
using Distributions

# Doctest setup
DocMeta.setdocmeta!(DynamicPPL, :DocTestSetup, :(using DynamicPPL); recursive=true)

makedocs(;
    sitename="DynamicPPL",
    format=Documenter.HTML(),
    modules=[DynamicPPL],
    pages=[
        "Home" => "index.md",
        "API" => "api.md",
        "Tutorials" => ["tutorials/prob-interface.md"],
    ],
    checkdocs=:exports,
)

deploydocs(; repo="github.com/TuringLang/DynamicPPL.jl.git", push_preview=true)
