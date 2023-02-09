#pragma once

#include "cstdint"

#include <iostream>
#include <string>
#include <type_traits>
#include <utility>

#ifdef __clang__

#define FORCEINLINE [[clang::always_inline]]

#elif defined(__MSVC__)

#define FORCEINLINE __forceinline

#elif defined(__GNUC__)

#define FORCEINLINE __attribute__((always_inline))

#endif

#define NODISCARD [[nodiscard]]

template<typename T>
struct STRemoveRef
{
	using Type = T;
	using ConstType = const T;
};


template<bool Flag, typename Arg = void>
struct STEnableIf
{
};

template<typename Arg>
struct STEnableIf<true, Arg>
{
};

template<typename T>
using TRemoveRef = typename STRemoveRef<T>::Type;

template<typename T, T... Indices>
struct STIntegerSequenceWrapper
{
};

template<typename T, T Size>
using TMakeIntegerSequence = __make_integer_seq<STIntegerSequenceWrapper, T, Size>;

template<typename T>
NODISCARD constexpr T&& Move(TRemoveRef<T>&& Arg) noexcept
{
	return static_cast<TRemoveRef<T>&&>(Arg);
}

template<typename T>
NODISCARD constexpr T&& Forward(TRemoveRef<T>& Arg) noexcept
{
	return static_cast<T&&>(Arg);
}

using TString = std::string;

using int32 = int32_t;
using int64 = int64_t;

using uint32 = uint32_t;
using uint64 = uint64_t;
using uint16 = uint16_t;

using uint8 = uint8_t;
using int8 = int8_t;
