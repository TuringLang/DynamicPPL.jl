function DynamicPPL.setval!(
    vi::DynamicPPL.VarInfo, data::InferenceObjects.Dataset, draw_id::Int, chain_id::Int
)
    return DynamicPPL.setval!(vi, data[draw=draw_id, chain=chain_id])
end

function DynamicPPL.setval_and_resample!(
    vi::DynamicPPL.VarInfoOrThreadSafeVarInfo,
    data::InferenceObjects.Dataset,
    draw_id::Int,
    chain_id::Int,
)
    return DynamicPPL.setval_and_resample!(vi, data[draw=draw_id, chain=chain_id])
end
