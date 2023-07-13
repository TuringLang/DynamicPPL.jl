#### prepare the models and chains for testing ####
# (1) manually create a chain using MCMCChains - we know what parameter names are in the chain
val = [1 2; 3 4; 5 6; 7 8; 9 10; 11 12; 13 14; 15 16; 17 18; 19 20]
val = Matrix{Int}(val)
chn_1 = Chains(val, [:s, :m])

# (2) sample a Turing model to create a chain
@model function gdemo(x)
    mu ~ MvNormal([0, 0, 0], [1 0 0; 0 1 0; 0 0 1])
    return x ~ MvNormal(mu, [1 0 0; 0 1 0; 0 0 1])
end
model_2 = gdemo([0, 0, 0]) 
chn_2 = sample(model_2, NUTS(), 100) 

#### test the functions in model_utils.jl ####
@testset "model_utils.jl" begin
    @testset "varname_in_chain" begin
        @test all(values(vn_in_chain(VarName(:s), chn_1, 1, 1, OrderedDict())))
        @test all(values(vn_in_chain(VarName(:m), chn_1, 1, 1, OrderedDict())))
        @test all(values(vn_in_chain(VarName(Symbol("mu[1]")), chn_2, 1, 1, OrderedDict())))

        @test all(values(varname_in_chain!(VarInfo(model_2)[VarName(:mu)], VarName(:mu), chn_2, 1, 1, OrderedDict())))

        @test length(keys(varname_in_chain!(VarInfo(model_2), VarName(:mu), chn_2, 1, 1, OrderedDict())))==3
        @test all(values(varname_in_chain!(VarInfo(model_2), VarName(:mu), chn_2, 1, 1, OrderedDict())))

        @test length(keys(varname_in_chain!(model_2, VarName(:mu), chn_2, 1, 1, OrderedDict())))==3
        @test all(values(varname_in_chain!(model_2, VarName(:mu), chn_2, 1, 1, OrderedDict())))

        @test varname_in_chain(VarInfo(model_2), VarName(:mu), chn_2, 1, 1)
        @test varname_in_chain(model_2, VarName(:mu), chn_2, 1, 1)

    end
    @testset "varnames_in_chain" begin
        @test length(varnames_in_chain!(VarInfo(model_2), chn_2, OrderedDict())) == 3
        @test all(values(varnames_in_chain!(VarInfo(model_2), chn_2, OrderedDict())))
        @test length(varnames_in_chain!(model_2, chn_2, OrderedDict())) == 3
        @test all(values(varnames_in_chain!(model_2, chn_2, OrderedDict())))

        @test varnames_in_chain(model_2, chn_2)
    end
    @testset "values_from_chain" begin
        @test isa(values_from_chain(VarName(:s), chn_1, 1, 1), Number)
        @test isa(values_from_chain(VarName(Symbol("mu[1]")), chn_2, 1, 1), Number)

        @test all(isa.(values_from_chain(VarInfo(model_2)[VarName(:mu)], VarName(:mu), chn_2, 1, 1), Number))

        @test all(isa.(values_from_chain(VarInfo(model_2), VarName(:mu), chn_2, 1, 1), Number))

        @test all(isa.(collect(values(values_from_chain!(model_2, chn_2, 1, 1, OrderedDict())))[1], Number))
        
        @test all(isa.(collect(values(values_from_chain!(VarInfo(model_2), chn_2, 1, 1, OrderedDict())))[1], Number))
    end
    @testset "value_iterator_from_chain" begin
        all_values = collect(value_iterator_from_chain(model_2, chn_2))
        for ordered_dict in all_values
            @test all(isa.(collect(values(ordered_dict))[1], Number))
        end
    end
end