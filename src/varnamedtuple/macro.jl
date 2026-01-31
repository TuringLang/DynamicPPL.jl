"""
    @vnt begin ... end

Construct a `VarNamedTuple` from a block of assignments. This is best illustrated by
example:

```jldoctest
julia> using DynamicPPL

julia> @vnt begin
           a = 1
           b = 2
       end
VarNamedTuple
├─ a => 1
└─ b => 2
```

You can set entirely arbitrary variables:

```jldoctest; setup=:(using DynamicPPL)
julia> @vnt begin
           a.b.c.d.e = "hello"
       end
VarNamedTuple
└─ a => VarNamedTuple
        └─ b => VarNamedTuple
                └─ c => VarNamedTuple
                        └─ d => VarNamedTuple
                                └─ e => "hello"
```

For variables that have indexing, it is often necessary to provide a template, so that the
VNT 'knows' what kind of array is being used to store the values, and can set the values in
the appropriate places (consider e.g. OffsetArrays where `x[1]` may not mean what it usually
does).

This is done by inserting a `@template` macro call in the block. The top-level symbols must
already be defined in the current scope. For example:

```jldoctest; setup=:(using DynamicPPL)
julia> x = zeros(5); y = zeros(3, 3);

julia> @vnt begin
            @template x y
            x[1] = 1.0
            y[1, 1] = 2.0
       end
VarNamedTuple
├─ x => PartialArray size=(5,) data::Vector{Float64}
│       └─ (1,) => 1.0
└─ y => PartialArray size=(3, 3) data::Matrix{Float64}
        └─ (1, 1) => 2.0
```

If no template is provided, the VNT will use a `GrowableArray`. This can produce correct
results in simple cases, but is not recommended for general use. Please see the
VarNamedTuple documentation for more details.

```jldoctest; setup=:(using DynamicPPL)
julia> @vnt begin
            # No template provided.
            x[1] = 1.0
            y[1, 1] = 2.0
       end
VarNamedTuple
├─ x => PartialArray size=(1,) data::DynamicPPL.VarNamedTuples.GrowableArray{Float64, 1}
│       └─ (1,) => 1.0
└─ y => PartialArray size=(1, 1) data::DynamicPPL.VarNamedTuples.GrowableArray{Float64, 2}
        └─ (1, 1) => 2.0
```
"""
macro vnt(expr)
    return _vnt(expr)
end

# This is copy pasted from DynamicPPL src/compiler.jl. It's duplicated here because VNT code
# might eventually be moved to AbstractPPL, and we don't want to have to depend on
# DynamicPPL stuff there.
_get_top_sym(expr::Symbol) = expr
function _get_top_sym(expr::Expr)
    if Meta.isexpr(expr, :ref)
        return _get_top_sym(expr.args[1])
    elseif Meta.isexpr(expr, :.)
        return _get_top_sym(expr.args[1])
    else
        error("unreachable")
    end
end

function _vnt(input)
    Meta.isexpr(input, :block) ||
        error("`@vnt` expects a block expression (e.g. `@vnt begin ... end`)")
    @gensym vnt
    templated_symbols = Set{Symbol}()
    output = Expr(:block)
    push!(output.args, :($vnt = VarNamedTuple()))
    for expr in input.args
        if expr isa LineNumberNode
            push!(output.args, expr)
        elseif Meta.isexpr(expr, :macrocall) && expr.args[1] == Symbol("@template")
            # Extract all the templated symbols
            for arg in expr.args[2:end]
                if arg isa LineNumberNode
                    continue
                elseif arg isa Symbol
                    push!(templated_symbols, arg)
                else
                    error(
                        "`@template` only accepts top-level symbols as arguments, separated by swhitespace (e.g. `@template x y z`)",
                    )
                end
            end
        elseif Meta.isexpr(expr, :(=))
            lhs, rhs = expr.args
            vn = AbstractPPL.varname(lhs, false)
            top_level_sym = _get_top_sym(lhs)
            if top_level_sym in templated_symbols
                push!(
                    output.args,
                    :(
                        $vnt = DynamicPPL.templated_setindex!!(
                            $vnt, $rhs, $vn, $(esc(top_level_sym))
                        )
                    ),
                )
            else
                push!(output.args, :($vnt = BangBang.setindex!!($vnt, $rhs, $vn)))
            end
        else
            error("unexpected expression in `@vnt` block: $expr")
        end
    end
    push!(output.args, :($vnt))
    return output
end
