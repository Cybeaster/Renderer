#pragma once
#include <tuple>
#include "Tuple/Tuple.hpp"

namespace RenderAPI
{
    template <typename... Types>
    using TTuple = TTElemSequenceTuple<Types...>;
}
