@testset "tag" begin
    for chunksize in (0, 1, 10)
        ad = ADTypes.AutoForwardDiff(; chunksize=chunksize)
        standardtag = if !isdefined(Base, :get_extension)
            DynamicPPL.DynamicPPLForwardDiffExt.standardtag
        else
            Base.get_extension(DynamicPPL, :DynamicPPLForwardDiffExt).standardtag
        end
        @test standardtag(ad)
        for tag in (false, 0, 1)
            @test !standardtag(AutoForwardDiff(; chunksize=chunksize, tag=tag))
        end
    end
end
