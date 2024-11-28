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

function DynamicPPL.determine_varinfo(
    model::DynamicPPL.Model,
    context::DynamicPPL.AbstractContext=DynamicPPL.DefaultContext();
    verbose::Bool=false,
    only_tilde::Bool=true
)
    # First we try with the typed varinfo.
    varinfo = DynamicPPL.typed_varinfo(model)
    issuccess = true

    # Let's make sure that both evaluation and sampling doesn't result in type errors.
    f_eval, argtypes_eval = DynamicPPL.DebugUtils.gen_evaluator_call_with_types(
        model, varinfo, context
    )
    result_eval = JET.report_call(f_eval, argtypes_eval)
    reports_eval = JET.get_reports(result_eval)
    if only_tilde
        reports_eval = filter(report_has_error_in_tilde, reports_eval)
    end
    # If we get reports => we had issues so we use the untyped varinfo.
    issuccess &= length(reports_eval) == 0
    if issuccess
        # Evaluation succeeded, let's try sampling.
        f_sample, argtypes_sample = DynamicPPL.DebugUtils.gen_evaluator_call_with_types(
            model, varinfo, DynamicPPL.SamplingContext(context)
        )
        result_sample = JET.report_call(f_sample, argtypes_sample)
        reports_sample = JET.get_reports(result_sample)
        if only_tilde
            reports_sample = filter(report_has_error_in_tilde, reports_sample)
        end
        # If we get reports => we had issues so we use the untyped varinfo.
        issuccess &= length(reports_sample) == 0
        if !issuccess && verbose
            # Show the user the issues.
            @warn "Sampling with typed varinfo failed with the following issues:"
            for report in reports_sample
                @warn report
            end
        end
    elseif verbose
        # Show the user the issues.
        @warn "Evaluaton with typed varinfo failed with the following issues:"
        for report in reports_eval
            @warn report
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
 
