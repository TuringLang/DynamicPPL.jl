using DynamicPPL
using BenchmarkTools

import Weave
import Markdown

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
    doc = Weave.WeaveDoc(indoc, nothing)
    doc = Weave.run_doc(doc, doctype = "github", mod = mod, args = args, kwargs...)
    rendered = Weave.render_doc(doc)
    return display(Markdown.parse(rendered))
end
