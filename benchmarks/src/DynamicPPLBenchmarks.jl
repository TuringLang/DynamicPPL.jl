module DynamicPPLBenchmarks

using DynamicPPL
using BenchmarkTools

import Weave
import Markdown

import LibGit2, Pkg

export weave_benchmarks

function time_model_def(model_def, args...)
    return @time model_def(args...)
end

function benchmark_untyped_varinfo!(suite, m)
    vi = VarInfo()
    # Populate.
    m(vi)
    # Evaluate.
    suite["evaluation_untyped"] = @benchmarkable $m($vi)
    return suite
end

function benchmark_typed_varinfo!(suite, m)
    # Populate.
    vi = VarInfo(m)
    # Evaluate.
    suite["evaluation_typed"] = @benchmarkable $m($vi)
    return suite
end

function typed_code(m, vi = VarInfo(m))
    rng = DynamicPPL.Random.MersenneTwister(42);
    spl = DynamicPPL.SampleFromPrior()
    ctx = DynamicPPL.SamplingContext(rng, spl, DynamicPPL.DefaultContext())

    results = code_typed(m.f, Base.typesof(m, vi, ctx, m.args...))
    return first(results)
end

function make_suite(m)
    suite = BenchmarkGroup()
    benchmark_untyped_varinfo!(suite, m)
    benchmark_typed_varinfo!(suite, m)

    return suite
end

function weave_child(indoc; mod, args, kwargs...)
    # FIXME: Make this work for other output formats than just `github`.
    doc = Weave.WeaveDoc(indoc, nothing)
    doc = Weave.run_doc(doc, doctype = "github", mod = mod, args = args, kwargs...)
    rendered = Weave.render_doc(doc)
    return display(Markdown.parse(rendered))
end

function pkgversion(m::Module)
    projecttoml_path = joinpath(dirname(pathof(m)), "..", "Project.toml")
    return Pkg.TOML.parsefile(projecttoml_path)["version"]
end

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

function weave_benchmarks(
    ;
    benchmarkbody=joinpath(dirname(pathof(DynamicPPLBenchmarks)), "..", "benchmark_body.jmd"),
    include_commit_id=false,
    name=default_name(include_commit_id=include_commit_id),
    name_old=nothing,
    include_typed_code=false,
    doctype="github",
    outpath="results/$(name)/",
    kwargs...
)
    args = Dict(
        :benchmarkbody => benchmarkbody,
        :name => name,
        :include_typed_code => include_typed_code
    )
    if !isnothing(name_old)
        args[:name_old] = name_old
    end
    @info "Storing output in $(outpath)"
    mkpath(outpath)
    Weave.weave("benchmarks.jmd", doctype; out_path=outpath, args=args, kwargs...)
end

end # module
