module DynamicPPLJETExt

using DynamicPPL: DynamicPPL
using JET: JET

"""
    is_tilde_instance(x)

Return `true` if `x` is a method instance of a tilde function, otherwise `false`.
"""
is_tilde_instance(x) = false
is_tilde_instance(frame::JET.VirtualFrame) = is_tilde_instance(frame.linfo)
is_tilde_instance(mi::Core.MethodInstance) = is_tilde_instance(mi.specTypes.parameters[1])
is_tilde_instance(::Type{typeof(DynamicPPL.tilde_assume!!)}) = true
is_tilde_instance(::Type{typeof(DynamicPPL.tilde_observe!!)}) = true
is_tilde_instance(::Type{typeof(DynamicPPL.dot_tilde_assume!!)}) = true
is_tilde_instance(::Type{typeof(DynamicPPL.dot_tilde_observe!!)}) = true

"""
    report_has_error_in_tilde(report)

Return `true` if the given error `report` contains a tilde function in its frames, otherwise `false`.

This is used to filter out reports that occur outside of the tilde pipeline, in an attempt to avoid
warning the user about DynamicPPL doing something wrong when it is in fact an issue with the user's code.
"""
function report_has_error_in_tilde(report)
    frames = report.vst
    return any(is_tilde_instance, frames)
end

function DynamicPPL.is_suitable_varinfo(
    model::DynamicPPL.Model,
    context::DynamicPPL.AbstractContext,
    varinfo::DynamicPPL.AbstractVarInfo;
    only_tilde::Bool=true,
)
    # Let's make sure that both evaluation and sampling doesn't result in type errors.
    f, argtypes = DynamicPPL.DebugUtils.gen_evaluator_call_with_types(
        model, varinfo, context
    )
    result = JET.report_call(f, argtypes)
    reports = JET.get_reports(result)
    # TODO: Should we use a more aggressive filter here?
    if only_tilde
        reports = filter(report_has_error_in_tilde, reports)
    end
    return length(reports) == 0, reports
end

function DynamicPPL._determine_varinfo_jet(
    model::DynamicPPL.Model,
    context::DynamicPPL.AbstractContext;
    only_tilde::Bool=true,
)
    # First we try with the typed varinfo.
    varinfo = if DynamicPPL.hassampler(context)
        # Don't need to add sampling context for this to work.
        DynamicPPL.typed_varinfo(model, context)
    else
        # Need a sampling context to initialize the varinfo.
        DynamicPPL.typed_varinfo(model, DynamicPPL.SamplingContext(context))
    end
    issuccess = true

    # Let's make sure that both evaluation and sampling doesn't result in type errors.
    issuccess, reports = DynamicPPL.is_suitable_varinfo(
        model, context, varinfo; only_tilde
    )

    if !issuccess
        # Useful information for debugging.
        @debug "Evaluaton with typed varinfo failed with the following issues:"
        for report in reports
            @debug report
        end
    end

    # If we didn't fail anywhere, we return the type stable one.
    return if issuccess
        varinfo
    else
        # Warn the user that we can't use the type stable one.
        @warn "Model seems incompatible with typed varinfo. Falling back to untyped varinfo."
        DynamicPPL.untyped_varinfo(model)
    end
end

end
