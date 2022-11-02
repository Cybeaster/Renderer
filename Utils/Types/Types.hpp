#pragma once

#include <string>
#include "cstdint"

template <typename T, T... Indices>
struct TIntegerSequenceWrapper
{
};

template<typename T, T Size>
using TTMakeIntegerSequence = __make_integer_seq<TIntegerSequenceWrapper,T, Size>;

using TString = std::string;

using int32 = int32_t;
using int64 = int64_t;

using uint32 = uint32_t;
using uint64 = uint64_t;
using uint16 = uint16_t;

using uint8 = uint8_t;
using int8 = int8_t;
