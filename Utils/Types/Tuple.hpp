#pragma once
#include "Tuple/Tuple.hpp"

#include <tuple>

namespace RAPI
{
template<typename... Types>
using OTuple = OTElemSequenceTuple<Types...>;
}
