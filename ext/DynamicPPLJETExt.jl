module DynamicPPLJETExt

using DynamicPPL: DynamicPPL
using JET: JET

function DynamicPPL.Experimental.is_suitable_varinfo(
    model::DynamicPPL.Model, varinfo::DynamicPPL.AbstractVarInfo; only_ddpl::Bool=true
)
    # Let's make sure that both evaluation and sampling doesn't result in type errors.
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
    # Use SamplingContext to test type stability.
    sampling_model = DynamicPPL.contextualize(
        model, DynamicPPL.SamplingContext(model.context)
    )

    # First we try with the typed varinfo.
    varinfo = DynamicPPL.typed_varinfo(sampling_model)

    # Let's make sure that both evaluation and sampling doesn't result in type errors.
    issuccess, result = DynamicPPL.Experimental.is_suitable_varinfo(
        sampling_model, varinfo; only_ddpl
    )

    if !issuccess
        # Useful information for debugging.
        @debug "Evaluaton with typed varinfo failed with the following issues:"
        @debug result
    end

    # If we didn't fail anywhere, we return the type stable one.
    return if issuccess
        varinfo
    else
        # Warn the user that we can't use the type stable one.
        @warn "Model seems incompatible with typed varinfo. Falling back to untyped varinfo."
        DynamicPPL.untyped_varinfo(sampling_model)
    end
end

end
