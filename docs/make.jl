using Documenter
using DynamicPPL
using Pluto: Configuration.CompilerOptions
using PlutoStaticHTML

const PACKAGE = DynamicPPL

# Doctest setup
DocMeta.setdocmeta!(PACKAGE, :DocTestSetup, :(using PACKAGE); recursive=true)

tutorials = [
    "The Basics",
]

const tutorials_dir = joinpath(pkgdir(PACKAGE), "docs", "src", "tutorials")

include("build.jl")

build()
md_files = markdown_files()
T = [t => f for (t, f) in zip(tutorials, md_files)]

DocMeta.setdocmeta!(PACKAGE, :DocTestSetup, :(using PACKAGE); recursive=true)

makedocs(;
    sitename="$PACKAGE",
    format=Documenter.HTML(),
    modules=[PACKAGE],
    pages=[
        "API" => "api.md", 
        "TestUtils" => "test_utils.md",
        "Tutorials" => T,
    ],
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
        r"ERROR: (LoadError:\s)+",
    ],
)

deploydocs(; repo="github.com/TuringLang/$PACKAGE.jl.git", push_preview=true)

# Useful for local development.
cd(pkgdir(PACKAGE))
