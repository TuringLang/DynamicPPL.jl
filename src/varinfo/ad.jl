function __init__()
    @require ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210" begin
        value(x::ForwardDiff.Dual) = ForwardDiff.value(x)
        value(x::AbstractArray{<:ForwardDiff.Dual}) = ForwardDiff.value.(x)
    end
    @require ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267" begin
        value(x::ReverseDiff.TrackedReal) = ReverseDiff.value(x)
        value(x::ReverseDiff.TrackedArray) = ReverseDiff.value(x)
        value(x::AbstractArray{<:ReverseDiff.TrackedReal}) = ReverseDiff.value.(x)
    end
    @require Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c" begin
        value(x::Tracker.TrackedReal) = Tracker.data(x)
        value(x::Tracker.TrackedArray) = Tracker.data(x)
        value(x::AbstractArray{<:Tracker.TrackedReal}) = Tracker.data.(x)
    end
end