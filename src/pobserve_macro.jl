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
        total_likelihoods = sum(fetch.(likelihood_tasks))
        # println("Total likelihoods: ", total_likelihoods)
        $(esc(:(__varinfo__))) = $(DynamicPPL.accloglikelihood!!)(
            $(esc(:(__varinfo__))), total_likelihoods
        )
        nothing
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
    transformed_statements = map(statements) do stmt
        # skip non-tilde statements
        # TODO: dot-tilde
        @capture(stmt, lhs_ ~ rhs_) || return :($(esc(stmt)))
        # if the above matched, we transform the tilde statement
        # TODO: We should probably perform some checks to make sure that this
        # indeed was meant to be an observe statement.
        :($loglike += $(Distributions.logpdf)($(esc(rhs)), $(esc(lhs))))
    end
    ending_statement = loglike
    new_statements = [beginning_statement, transformed_statements..., ending_statement]
    return Expr(:block, new_statements...)
end
