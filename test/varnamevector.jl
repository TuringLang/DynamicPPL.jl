@testset "VarNameVector" begin
    vns = [@varname(x[1]), @varname(x[2]), @varname(x[3])]
    vals = [1, 2:3, reshape(4:9, 2, 3)]
    vnv = VarNameVector(vns, vals)

    # `getindex`
    for (vn, val) in zip(vns, vals)
        @test vnv[vn] == val
    end

    # `setindex!`
    for (vn, val) in zip(vns, vals)
        vnv[vn] = val .+ 100
    end

    for (vn, val) in zip(vns, vals)
        @test vnv[vn] == val .+ 100
    end
end
