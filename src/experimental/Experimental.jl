module Experimental

using ..DynamicPPL: DynamicPPL

"""
    evaluatortype(f)
    evaluatortype(f, nargs)
    evaluatortype(f, argtypes)
    evaluatortype(m::DynamicPPL.Model)

Returns the evaluator-type for model `m` or a model-constructor `f`.

(!!!) If you're using Revise.jl, remember that you might need to re-instaniate
the model since `evaluatortype` might have changed.
"""
function evaluatortype(f, argtypes)
    rets = Core.Compiler.return_types(f, argtypes)
    if (length(rets) != 1) || !(first(rets) <: DynamicPPL.Model)
        error("inferred return-type of $(f) using $(argtypes) is not `Model`; please specify argument types")
    end
    # Extract the anonymous evaluator.
    return first(rets).parameters[1]
end
evaluatortype(f, nargs::Int) = evaluatortype(f, ntuple(_ -> Missing, nargs))
function evaluatortype(f)
    m = first(methods(f))
    # Extract the arguments (first element is the method itself).
    nargs = length(m.sig.parameters) - 1

    return evaluatortype(f, nargs)
end
evaluatortype(::DynamicPPL.Model{F}) where {F} = F

end
