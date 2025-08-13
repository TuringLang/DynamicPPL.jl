module DynamicPPLJETExt

using DynamicPPL: DynamicPPL
using JET: JET

function DynamicPPL.Experimental.is_suitable_varinfo(
    model::DynamicPPL.Model, varinfo::DynamicPPL.AbstractVarInfo; only_ddpl::Bool=true
)
    f, argtypes = DynamicPPL.DebugUtils.gen_evaluator_call_with_types(model, varinfo)
    # If specified, we only check errors originating somewhere in the DynamicPPL.jl.
    # This way we don't just fall back to untyped if the user's code is the issue.
    result = if only_ddpl
        JET.report_call(f, argtypes; target_modules=(JET.AnyFrameModule(DynamicPPL),))
    else
        JET.report_call(f, argtypes)
    end
    return length(JET.get_reports(result)) == 0, result
end

function DynamicPPL.Experimental._determine_varinfo_jet(
    model::DynamicPPL.Model; only_ddpl::Bool=true
)
    # Generate a typed varinfo to test model type stability with
    varinfo = DynamicPPL.typed_varinfo(model)

    # Check type stability of evaluation (i.e. DefaultContext)
    model = DynamicPPL.contextualize(
        model, DynamicPPL.setleafcontext(model.context, DynamicPPL.DefaultContext())
    )
    eval_issuccess, eval_result = DynamicPPL.Experimental.is_suitable_varinfo(
        model, varinfo; only_ddpl
    )
    if !eval_issuccess
        @debug "Evaluation with typed varinfo failed with the following issues:"
        @debug eval_result
    end

    # Check type stability of initialisation (i.e. InitContext)
    model = DynamicPPL.contextualize(
        model, DynamicPPL.setleafcontext(model.context, DynamicPPL.InitContext())
    )
    init_issuccess, init_result = DynamicPPL.Experimental.is_suitable_varinfo(
        model, varinfo; only_ddpl
    )
    if !init_issuccess
        @debug "Initialisation with typed varinfo failed with the following issues:"
        @debug init_result
    end

    # If neither of them failed, we can return the typed varinfo as it's type stable.
    return if (eval_issuccess && init_issuccess)
        varinfo
    else
        # Warn the user that we can't use the type stable one.
        @warn "Model seems incompatible with typed varinfo. Falling back to untyped varinfo."
        DynamicPPL.untyped_varinfo(model)
    end
end

end
