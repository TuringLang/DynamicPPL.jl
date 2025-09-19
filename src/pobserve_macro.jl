using MacroTools: @capture, @q

"""
    @pobserve

Perform observations in parallel.
"""
macro pobserve(expr)
    return _pobserve(expr)
end

function _pobserve(expr::Expr)
    @capture(
        expr,
        for ctr_ in iterable_
            block_
        end
    ) || error("expected for loop")
    # reconstruct the for loop with the processed block
    return_expr = @q begin
        likelihood_tasks = map($(esc(iterable))) do $(esc(ctr))
            Threads.@spawn begin
                $(process_tilde_statements(block))
            end
        end
        retvals_and_likelihoods = fetch.(likelihood_tasks)
        total_likelihoods = sum(last, retvals_and_likelihoods)
        # println("Total likelihoods: ", total_likelihoods)
        $(esc(:(__varinfo__))) = $(DynamicPPL.accloglikelihood!!)(
            $(esc(:(__varinfo__))), total_likelihoods
        )
        map(first, retvals_and_likelihoods)
    end
    return return_expr
end

"""
    process_tilde_statements(expr)

This function traverses a block expression `expr` and transforms any
lines in it that look like `lhs ~ rhs` into a simple accumulation of
likelihoods, i.e., `Distributions.logpdf(rhs, lhs)`.
"""
function process_tilde_statements(expr::Expr)
    @capture(
        expr,
        begin
            statements__
        end
    ) || error("expected block")
    @gensym loglike
    beginning_statement =
        :($loglike = zero($(DynamicPPL.getloglikelihood)($(esc(:(__varinfo__))))))
    n_statements = length(statements)
    transformed_statements::Vector{Vector{Expr}} = map(enumerate(statements)) do (i, stmt)
        is_last = i == n_statements
        if @capture(stmt, lhs_ ~ rhs_)
            # TODO: We should probably perform some checks to make sure that this
            # indeed was meant to be an observe statement.
            @gensym left
            e = quote
                $left = $(esc(lhs))
                $loglike += $(Distributions.logpdf)($(esc(rhs)), $left)
            end
            is_last && push!(e.args, :(($left, $loglike)))
            e.args
        elseif @capture(stmt, lhs_ .~ rhs_)
            @gensym val
            e = [
                # TODO: dot-tilde
                :($val = $(esc(stmt))),
            ]
            is_last && push!(e, :(($val, $loglike)))
            e
        else
            @gensym val
            e = [:($val = $(esc(stmt)))]
            is_last && push!(e, :(($val, $loglike)))
            e
        end
    end
    new_statements = [beginning_statement, reduce(vcat, transformed_statements)...]
    return Expr(:block, new_statements...)
end
