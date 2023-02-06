#pragma once

#include "cstdint"

#include <iostream>
#include <string>

#ifdef __clang__

#define FORCEINLINE [[clang::always_inline]]

#elif defined(__MSVC__)

#define FORCEINLINE __forceinline

#elif defined(__GNUC__)

#define FORCEINLINE __attribute__((always_inline))

#endif

#define NODISCARD [[nodiscard]]

template<typename T, T... Indices>
struct SIntegerSequenceWrapper
{
};

template<typename T, T Size>
using TTMakeIntegerSequence = __make_integer_seq<SIntegerSequenceWrapper, T, Size>;

template<typename T>
T&& Move(T Arg)
{
	return static_cast<T&&>(Arg);
}

template<typename T>
T&& Forward(T Arg)
{
	return std::forward(Arg);
}

using TString = std::string;

using int32 = int32_t;
using int64 = int64_t;

using uint32 = uint32_t;
using uint64 = uint64_t;
using uint16 = uint16_t;

using uint8 = uint8_t;
using int8 = int8_t;
