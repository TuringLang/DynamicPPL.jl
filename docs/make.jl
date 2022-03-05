using Documenter
using DynamicPPL

# Doctest setup
DocMeta.setdocmeta!(DynamicPPL, :DocTestSetup, :(using DynamicPPL); recursive=true)

makedocs(;
    sitename="DynamicPPL",
    format=Documenter.HTML(),
    modules=[DynamicPPL],
    pages=["Home" => "index.md", "TestUtils" => "test_utils.md"],
    strict=true,
    checkdocs=:exports,
)

deploydocs(; repo="github.com/TuringLang/DynamicPPL.jl.git", push_preview=true)
