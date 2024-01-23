@testset "tag" begin
    for chunksize in (0, 1, 10)
        ad = Turing.AutoForwardDiff(; chunksize=chunksize)
        @test ad === Turing.AutoForwardDiff(; chunksize=chunksize)
        @test Turing.Essential.standardtag(ad)
        for standardtag in (false, 0, 1)
            @test !Turing.Essential.standardtag(Turing.AutoForwardDiff(; chunksize=chunksize, tag=standardtag))
        end
    end
end

