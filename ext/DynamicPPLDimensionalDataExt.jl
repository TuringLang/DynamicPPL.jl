module DynamicPPLDimensionalDataExt
using DynamicPPL: DynamicPPL
using DimensionalData: DimArray

DynamicPPL._partial_binding_array(value::DimArray) = parent(value) isa Array
end
