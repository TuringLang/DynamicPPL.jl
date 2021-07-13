module DynamicPPLBenchmarks

using DynamicPPL
using BenchmarkTools

using Weave: Weave
using Markdown: Markdown

using LibGit2: LibGit2
using Pkg: Pkg

export weave_benchmarks

function time_model_def(model_def, args...)
    return @time model_def(args...)
end

function benchmark_untyped_varinfo!(suite, m)
    vi = VarInfo()
    # Populate.
    m(vi)
    # Evaluate.
    suite["evaluation_untyped"] = @benchmarkable $m($vi, $(DefaultContext()))
    return suite
end

function benchmark_typed_varinfo!(suite, m)
    # Populate.
    vi = VarInfo(m)
    # Evaluate.
    suite["evaluation_typed"] = @benchmarkable $m($vi, $(DefaultContext()))
    return suite
end

function typed_code(m, vi=VarInfo(m))
    rng = DynamicPPL.Random.MersenneTwister(42)
    spl = DynamicPPL.SampleFromPrior()
    ctx = DynamicPPL.SamplingContext(rng, spl, DynamicPPL.DefaultContext())

    results = code_typed(m.f, Base.typesof(m, vi, ctx, m.args...))
    return first(results)
end

"""
    make_suite(model)

Create default benchmark suite for `model`.
"""
function make_suite(model)
    suite = BenchmarkGroup()
    benchmark_untyped_varinfo!(suite, model)
    benchmark_typed_varinfo!(suite, model)

    return suite
end

"""
    weave_child(indoc; mod, args, kwargs...)

Weave `indoc` with scope of `mod` into markdown.

Useful for weaving within weaving, e.g.
```julia
weave_child(child_jmd_path, mod = @__MODULE__, args = WEAVE_ARGS)
```
together with `results="markup"` and `echo=false` will simply insert
the weaved version of `indoc`.

# Notes
- Currently only supports `doctype == "github"`. Other outputs are "supported"
  in the sense that it works but you might lose niceties such as syntax highlighting.
"""
function weave_child(indoc; mod, args, kwargs...)
    # FIXME: Make this work for other output formats than just `github`.
    doc = Weave.WeaveDoc(indoc, nothing)
    doc = Weave.run_doc(doc; doctype="github", mod=mod, args=args, kwargs...)
    rendered = Weave.render_doc(doc)
    return display(Markdown.parse(rendered))
end

"""
    pkgversion(m::Module)

Return version of module `m` as listed in its Project.toml.
"""
function pkgversion(m::Module)
    projecttoml_path = joinpath(dirname(pathof(m)), "..", "Project.toml")
    return Pkg.TOML.parsefile(projecttoml_path)["version"]
end

"""
    default_name(; include_commit_id=false)

Construct a name from either repo information or package version
of `DynamicPPL`.

If the path of `DynamicPPL` is a git-repo, return name of current branch,
joined with the commit id if `include_commit_id` is `true`.

If path of `DynamicPPL` is _not_ a git-repo, it is assumed to be a release,
resulting in a name of the form `release-VERSION`.
"""
function default_name(; include_commit_id=false)
    dppl_path = abspath(joinpath(dirname(pathof(DynamicPPL)), ".."))

    # Extract branch name and commit id
    local name
    try
        githead = LibGit2.head(LibGit2.GitRepo(dppl_path))
        branchname = LibGit2.shortname(githead)

        name = replace(branchname, "/" => "_")
        if include_commit_id
            gitcommit = LibGit2.peel(LibGit2.GitCommit, githead)
            commitid = string(LibGit2.GitHash(gitcommit))
            name *= "-$(commitid)"
        end
    catch e
        if e isa LibGit2.GitError
            @info "No git repo found for $(dppl_path); extracting name from package version."
            name = "release-$(pkgversion(DynamicPPL))"
        else
            rethrow(e)
        end
    end

    return name
end

"""
    weave_benchmarks(input="benchmarks.jmd"; kwargs...)

Weave benchmarks present in `benchmarks.jmd` into a single file.

# Keyword arguments
- `benchmarkbody`: JMD-file to be rendered for each model.
- `include_commit_id=false`: specify whether to include commit-id in the default name.
- `name`: the name of directory in `results/` to use as output directory.
- `name_old=nothing`: if specified, comparisons of current run vs. the run pinted to
  by `name_old` will be included in the generated document.
- `include_typed_code=false`: if `true`, output of `code_typed` for the evaluator
  of the model will be included in the weaved document.
- Rest of the passed `kwargs` will be passed on to `Weave.weave`.
"""
function weave_benchmarks(
    input=joinpath(dirname(pathof(DynamicPPLBenchmarks)), "..", "benchmarks.jmd");
    benchmarkbody=joinpath(
        dirname(pathof(DynamicPPLBenchmarks)), "..", "benchmark_body.jmd"
    ),
    include_commit_id=false,
    name=default_name(; include_commit_id=include_commit_id),
    name_old=nothing,
    include_typed_code=false,
    doctype="github",
    outpath="results/$(name)/",
    kwargs...,
)
    args = Dict(
        :benchmarkbody => benchmarkbody,
        :name => name,
        :include_typed_code => include_typed_code,
    )
    if !isnothing(name_old)
        args[:name_old] = name_old
    end
    @info "Storing output in $(outpath)"
    mkpath(outpath)
    return Weave.weave(input, doctype; out_path=outpath, args=args, kwargs...)
end

end # module
