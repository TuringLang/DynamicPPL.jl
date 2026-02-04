module AquaTests

using Dates: now
@info "Testing $(@__FILE__)..."
__now__ = now()

using Aqua: Aqua
using DynamicPPL

Aqua.test_all(DynamicPPL)

@info "Completed $(@__FILE__) in $(now() - __now__)."

end # module
