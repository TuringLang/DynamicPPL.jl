using Documenter
using DocumenterInterLinks
using DynamicPPL
using AbstractPPL
# NOTE: This is necessary to ensure that if we print something from
# Distributions.jl in a doctest, then the shown value will not include
# a qualifier; that is, we don't want `Distributions.Normal{Float64}`
# but rather `Normal{Float64}`. The latter is what will then be printed
# in the doctest as run in `test/runtests.jl`, and so we need to stay
# consistent with that.
using Distributions
using DocumenterMermaid
# load MCMCChains package extension to make `predict` available
using MCMCChains
using AbstractMCMC: AbstractMCMC
using MarginalLogDensities: MarginalLogDensities

# Need this to document a method which uses a type inside the extension...
DPPLMLDExt = Base.get_extension(DynamicPPL, :DynamicPPLMarginalLogDensitiesExt)

# Doctest setup
DocMeta.setdocmeta!(
    DynamicPPL, :DocTestSetup, :(using DynamicPPL, MCMCChains); recursive=true
)

links = InterLinks("AbstractPPL" => "https://turinglang.org/AbstractPPL.jl/stable/")

makedocs(;
    sitename="DynamicPPL",
    # The API index.html page is fairly large, and violates the default HTML page size
    # threshold of 200KiB, so we double that.
    format=Documenter.HTML(;
        size_threshold=2^10 * 400, mathengine=Documenter.HTMLWriter.MathJax3()
    ),
    modules=[
        DynamicPPL,
        Base.get_extension(DynamicPPL, :DynamicPPLMCMCChainsExt),
        Base.get_extension(DynamicPPL, :DynamicPPLMarginalLogDensitiesExt),
    ],
    pages=[
        "Home" => "index.md",
        "Conditioning and fixing" => "conditionfix.md",
        "API" => "api.md",
        "VarNamedTuple" => [
            "vnt/motivation.md",
            "vnt/design.md",
            "vnt/implementation.md",
            "vnt/arraylikeblocks.md",
        ],
        "Initialisation strategies" => "init.md",
        "Transform strategies" => "transforms.md",
        "Accumulators" => "accumulators.md",
        "Model evaluation" => "flow.md",
        "Storing values" => "values.md",
    ],
    checkdocs=:exports,
    doctest=false,
    plugins=[links],
)
