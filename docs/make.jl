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
    doctestfilters=[
        # Older versions will show "0 element Array" instead of "Type[]".
        r"(Any\[\]|0-element Array{.+,[0-9]+})",
        # Older versions will show "Array{...,1}" instead of "Vector{...}".
        r"(Array{.+,\s?1}|Vector{.+})",
        # Older versions will show "Array{...,2}" instead of "Matrix{...}".
        r"(Array{.+,\s?2}|Matrix{.+})",
        # Errors from macros sometimes result in `LoadError: LoadError:`
        # rather than `LoadError:`, depending on Julia version.
        r"ERROR: LoadError: (LoadError:\s)?",
    ],
)

deploydocs(; repo="github.com/TuringLang/DynamicPPL.jl.git", push_preview=true)
