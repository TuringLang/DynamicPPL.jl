using Turing, Distributions, DynamicPPL, MCMCChains, Random, Test
Random.seed!(111)

#### prepare the models and chains for testing ####
# (1) manually create a chain using MCMCChains - we know what parameter names are in the chain
val = [1 2; 3 4; 5 6; 7 8; 9 10; 11 12; 13 14; 15 16; 17 18; 19 20]
val = Matrix{Int}(val)
chn_1 = Chains(val, [:s, :m])

# (2) sample a Turing model to create a chain
@model function gdemo(x)
    mu ~ MvNormal([0, 0, 0], [1 0 0; 0 1 0; 0 0 1])
    x ~ MvNormal(mu, [1 0 0; 0 1 0; 0 0 1])
end
model_2 = gdemo([0, 0, 0])  # provide an initial value for `x`
chn_2 = sample(model_2, NUTS(), 100) # NB: the parameter names in an MCMCChains can be retrieved using `namechn_2.name_map[:parameters]_map`: https://github.com/TuringLang/MCMCChains.jl/blob/master/src/chains.jl

# (3) sample the demo models in DynamicPPL to create a chain - we do not know what parameter names are in the chain beforehand 
model_3 = DynamicPPL.TestUtils.DEMO_MODELS[12]
var_info = VarInfo(model_3)
vns = DynamicPPL.TestUtils.varnames(model_3)
# generate a chain.
N = 100
vals_OrderedDict = mapreduce(hcat, 1:N) do _
    rand(OrderedDict, model_3)
end
vals_mat = mapreduce(hcat, 1:N) do i
    [vals_OrderedDict[i][vn] for vn in vns]
end
i = 1
for col in eachcol(vals_mat)
    col_flattened = []
    [push!(col_flattened, x...) for x in col]
    if i == 1
        chain_mat = Matrix(reshape(col_flattened, 1, length(col_flattened)))
    else
        chain_mat = vcat(chain_mat, reshape(col_flattened, 1, length(col_flattened)))
    end
    i += 1
end
chain_mat = convert(Matrix{Float64}, chain_mat)
# devise parameter names for chain
sample_values_vec = collect(values(vals_OrderedDict[1]))
symbol_names = []
chain_sym_map = Dict()
for k in 1:length(keys(var_info))
    vn_parent = keys(var_info)[k]
    sym = DynamicPPL.getsym(vn_parent)
    vn_children = DynamicPPL.varname_leaves(vn_parent, sample_values_vec[k])
    for vn_child in vn_children
        chain_sym_map[Symbol(vn_child)] = sym
        symbol_names = [symbol_names; Symbol(vn_child)]
    end
end
chn_3 = Chains(chain_mat, symbol_names)

#### test functions in model_utils.jl ####
@testset "model_utils.jl" begin
    @testset "varname_in_chain" begin
        # chn_1
        outputs = varname_in_chain(VarName(:s), chn_1, 1, 1)
        @test outputs[1] == true && outputs[2][:s] == true
        outputs = varname_in_chain(VarName(:m), chn_1, 1, 1)
        @test outputs[1] == true && outputs[2][:m] == true
        outputs = varname_in_chain(VarName(:x), chn_1, 1, 1)
        @test outputs[1] == false && isempty(outputs[2])

        # chn_2
        outputs = varname_in_chain(VarName(:mu), chn_2, 1, 1)
        @test outputs[1] == true && !isempty(outputs[2]) && all(values(outputs[2]) .== 1)

        # chn_3
        outputs = varname_in_chain(VarName(:a), chn_3, 1, 1)
        @test outputs[1] == false && isempty(outputs[2])
        outputs = varname_in_chain(VarName(symbol_names[1]), chn_3, 1, 1)
        @test !isempty(outputs[2]) && all(values(outputs[2]) .== 1) # note: all(values(outputs[2]) .== 1) is not enough - even an empty dictionary has all(values(outputs[2]) .== 1)
    end
    @testset "varnames_in_chain" begin
        outputs = varnames_in_chain(model_2, chn_2)
        @test outputs[1] == true && all(values(outputs[2][:mu]))
        outputs = varnames_in_chain(model_3, chn_3)
        @test outputs[1] == true
    end
    @testset "vn_values_from_chain" begin
        outputs = vn_values_from_chain(VarName(:mu), chn_2, 1, 1)
        @test outputs[1] == true && length(values(outputs[2])) == 3
        outputs = vn_values_from_chain(VarName(:s), chn_3, 1, 1)
        @test outputs[1] == true && length(values(outputs[2])) == 2
        outputs = vn_values_from_chain(VarName(Symbol("s[:,1][1]")), chn_3, 1, 2)
        @test outputs[1] == true
    end
    @testset "values_from_chain" begin
        output = values_from_chain(model_2, chn_2, 1, 1)
        @test length(output["chain_idx_1"]) == 3
        output = values_from_chain(model_3, chn_3, 1, 1)
        @test length(output["chain_idx_1"]) == 4

        output = values_from_chain(model_2, chn_2, 1, 1:10)
        @test length(output["chain_idx_1"]) == 3 &&
            all([length(vals) == 10 for vals in values(output["chain_idx_1"])])
        output = values_from_chain(model_3, chn_3, 1, 1:10)
        @test length(output["chain_idx_1"]) == 4 &&
            all([length(vals) == 10 for vals in values(output["chain_idx_1"])])
        output = values_from_chain(model_2, chn_2, 1, 1)
        @test length(output["chain_idx_1"]) == 3 &&
            all([length(vals) == 1 for vals in values(output["chain_idx_1"])])
        output = values_from_chain(model_2, chn_2, 1:1, 1)
        @test length(keys(output)) == 1 &&
            length(output["chain_idx_1"]) == 3 &&
            all([length(vals) == 1 for vals in values(output["chain_idx_1"])])
        output = values_from_chain(model_2, chn_2, 1, 1:1)
        @test length(output["chain_idx_1"]) == 3 &&
            all([length(vals) == 1 for vals in values(output["chain_idx_1"])])
        output = values_from_chain(model_2, chn_2, 1:1, 1:1)
        @test length(keys(output)) == 1 &&
            length(output["chain_idx_1"]) == 3 &&
            all([length(vals) == 1 for vals in values(output["chain_idx_1"])])

        output = values_from_chain(model_2, chn_2, nothing, nothing)
        @test length(keys(output)) == 1 &&
            length(output["chain_idx_1"]) == 3 &&
            all([length(vals) == 100 for vals in values(output["chain_idx_1"])])
        output = values_from_chain(model_3, chn_3, nothing, nothing)
        @test length(keys(output)) == 1 &&
            length(output["chain_idx_1"]) == 4 &&
            all([length(vals) == 100 for vals in values(output["chain_idx_1"])])
    end
end