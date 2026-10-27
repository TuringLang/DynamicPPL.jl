module DynamicPPLOffsetArraysExt
using DynamicPPL: DynamicPPL
using OffsetArrays: OffsetArray

DynamicPPL._partial_binding_array(value::OffsetArray) = parent(value) isa Array
end
