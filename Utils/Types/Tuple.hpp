#pragma once
#include "Tuple/Tuple.hpp"

#include <tuple>

namespace RenderAPI
{
template<typename... Types>
using OTuple = OTElemSequenceTuple<Types...>;
}
