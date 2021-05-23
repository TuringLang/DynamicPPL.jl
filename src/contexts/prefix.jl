struct PrefixContext{Prefix,C,LeafCtx} <: WrappedContext{LeafCtx}
    ctx::C

    function PrefixContext{Prefix}(ctx::AbstractContext) where {Prefix}
        return new{Prefix,typeof(ctx),typeof(ctx)}(ctx)
    end
    function PrefixContext{Prefix}(ctx::WrappedContext{LeafCtx}) where {Prefix,LeafCtx}
        return new{Prefix,typeof(ctx),LeafCtx}(ctx)
    end
end
PrefixContext{Prefix}() where {Prefix} = PrefixContext{Prefix}(EvaluationContext())

function rewrap(parent::PrefixContext{Prefix}, leaf::PrimitiveContext) where {Prefix}
    return PrefixContext{Prefix}(rewrap(childcontext(parent), leaf))
end

const PREFIX_SEPARATOR = Symbol(".")

function PrefixContext{PrefixInner}(
    ctx::PrefixContext{PrefixOuter}
) where {PrefixInner,PrefixOuter}
    if @generated
        :(PrefixContext{$(QuoteNode(Symbol(PrefixOuter, PREFIX_SEPARATOR, PrefixInner)))}(
            ctx.ctx
        ))
    else
        PrefixContext{Symbol(PrefixOuter, PREFIX_SEPARATOR, PrefixInner)}(ctx.ctx)
    end
end

function prefix(::PrefixContext{Prefix}, vn::VarName{Sym}) where {Prefix,Sym}
    if @generated
        return :(VarName{$(QuoteNode(Symbol(Prefix, PREFIX_SEPARATOR, Sym)))}(vn.indexing))
    else
        VarName{Symbol(Prefix, PREFIX_SEPARATOR, Sym)}(vn.indexing)
    end
end
