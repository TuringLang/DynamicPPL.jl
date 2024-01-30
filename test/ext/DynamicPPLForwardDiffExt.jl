@testset "tag" begin
    for chunksize in (0, 1, 10)
        ad = ADTypes.AutoForwardDiff(; chunksize=chunksize)
        forwarddiff_ext = Base.get_extension(DynamicPPL, :DynamicPPLForwardDiffExt)
        @test forwarddiff_ext.standardtag(ad)
        for standardtag in (false, 0, 1)
            @test !forwarddiff_ext.standardtag(
                AutoForwardDiff(; chunksize=chunksize, tag=standardtag)
            )
        end
    end
end
