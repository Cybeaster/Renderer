#pragma once
#include "TypeTraits.hpp"
#include "cstdint"

#include <iostream>
#include <string>
#include <type_traits>
#include <utility>


#ifdef NDEBUG

#ifdef __clang__

#define FORCEINLINE [[clang::always_inline]]

#elif defined(__MSVC__)

#define FORCEINLINE __forceinline

#elif defined(__GNUC__)

#define FORCEINLINE __attribute__((always_inline))

#endif

#else

#define FORCEINLINE inline

#endif

template<typename T>
NODISCARD constexpr TRemoveRef<T>&& Move(T&& Arg) noexcept
{
	return static_cast<TRemoveRef<T>&&>(Arg);
}

template<typename>
constexpr bool TIsLValueRef = false;

template<typename T>
constexpr bool TIsLValueRef<T&> = true;

template<typename T>
NODISCARD constexpr T&& Forward(TRemoveRef<T>& Arg) noexcept
{
	return static_cast<T&&>(Arg);
}

template<typename T>
NODISCARD constexpr T&& Forward(TRemoveRef<T>&& Arg) noexcept
{
	static_assert(TIsLValueRef<T>, "Bad forward call");
	return static_cast<T&&>(Arg);
}

using OString = std::string;