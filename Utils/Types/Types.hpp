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

#define REMOVE_COPY_FOR(ClassName)        \
	ClassName(const ClassName&) = delete; \
	ClassName& operator=(const ClassName&) = delete;

#define MIN_FLOAT (1.175494351e-38F) /* min positive value */
#define MAX_FLOAT (3.402823466e+38F)

#define MIN_DOUBLE (2.2250738585072014e-308) /* min positive value */
#define MAX_DOUBLE (1.7976931348623158e+308)

template<typename T>
struct STypeLimits;

template<>
struct STypeLimits<uint32>
{
	static uint32 Min()
	{
		return 0;
	}

	static uint32 MAX()
	{
		return UINT32_MAX;
	}
};

template<>
struct STypeLimits<float>
{
	static float Min()
	{
		return MIN_FLOAT;
	}

	static float MAX()
	{
		return MAX_FLOAT;
	}
};

template<>
struct STypeLimits<double>
{
	static double Min()
	{
		return MIN_DOUBLE;
	}

	static double MAX()
	{
		return MAX_DOUBLE;
	}
};

#define UINT32_INVALID_VALUE STypeLimits<uint32>::MAX()

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