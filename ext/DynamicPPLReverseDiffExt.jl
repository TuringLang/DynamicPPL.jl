module DynamicPPLReverseDiffExt

import LogDensityProblemsAD: ADgradient
using DynamicPPL: ADTypes, _make_ad_gradient, LogDensityFunction

ADgradient(ad::ADTypes.AutoReverseDiff, f::LogDensityFunction) = _make_ad_gradient(ad, f)

end # module
