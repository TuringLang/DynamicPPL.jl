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

This is done by inserting a `@template` macro call in the block. The `@template` macro
accepts whitespace-separated arguments, which must either be

- a single symbol (e.g. `@template x`), in which case the template is the value of `x`
  (and `x` must already be defined in the current scope); or
- an assignment of the form `y = expr`, in which case the template for `y` is the value of
  `expr`. In this case `y` does not need to be defined in the current scope, but any symbols
  referenced in `expr` must be.

For example:

```jldoctest; setup=:(using DynamicPPL)
julia> x = zeros(5); outside_y = zeros(3, 3);

julia> @vnt begin
            @template x y=outside_y
            x[1] = 1.0
            y[1, 1] = 2.0
       end
VarNamedTuple
├─ x => PartialArray size=(5,) data::Vector{Float64}
│       └─ (1,) => 1.0
└─ y => PartialArray size=(3, 3) data::Matrix{Float64}
        └─ (1, 1) => 2.0
```

!!! note
    You can use any expression in `@template y=expr`, even a function call that is
    completely contained within the macro (e.g. `@template y=zeros(3, 3)`). The macro makes
    sure that `expr` is only evaluated once, so there is no performance penalty to doing
    this.

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
    symbols_to_templates = Dict{Symbol,Expr}()
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
                    # e.g. @template x
                    if arg in keys(symbols_to_templates)
                        error("duplicate template definition for symbol: $arg")
                    end
                    symbols_to_templates[arg] = esc(arg)
                elseif Meta.isexpr(arg, :(=))
                    # e.g. @template y = x
                    sym, template_expr = arg.args
                    if sym in keys(symbols_to_templates)
                        error("duplicate template definition for symbol $sym")
                    end
                    # evaluate the template expression one time so that we don't
                    # reevaluate it every time we set a value in the VNT
                    new_sym = gensym()
                    push!(output.args, :($new_sym = $(esc(template_expr))))
                    symbols_to_templates[sym] = :(QuoteNode($new_sym))
                else
                    error("unexpected argument to `@template`: $arg")
                end
            end
        elseif Meta.isexpr(expr, :(=))
            lhs, rhs = expr.args
            vn = AbstractPPL.varname(lhs, false)
            top_level_sym = _get_top_sym(lhs)
            if top_level_sym in keys(symbols_to_templates)
                push!(
                    output.args,
                    :(
                        $vnt = DynamicPPL.templated_setindex!!(
                            $vnt, $rhs, $vn, $(symbols_to_templates[top_level_sym])
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
