#pragma once
#include "cstdint"

#include <iostream>
#include <string>
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

#define NODISCARD [[nodiscard]]

#define NODISC_FORCEINL NODISCARD FORCEINLINE
#define DELEGATE_NODISCARD [[nodiscard("Delegate's function result has to be stored in value!")]]

using int32 = int32_t;
using int64 = int64_t;

using uint32 = uint32_t;
using uint64 = uint64_t;
using uint16 = uint16_t;
using int16 = int16_t;

using uint8 = uint8_t;
using int8 = int8_t;

using char8 = char8_t;
using char16 = char16_t;
using char32 = char32_t;
using wchar = wchar_t;

using CWCharPTR = const wchar*;
using CCharPTR = const char*;
using OString = std::string;
#define INFINITE_LOOP_SCOPE() \
	while (true)

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
	constexpr static uint32 Min()
	{
		return 0;
	}

	constexpr static uint32 MAX()
	{
		return UINT32_MAX;
	}
};

template<>
struct STypeLimits<float>
{
	constexpr static float Min()
	{
		return MIN_FLOAT;
	}

	constexpr static float MAX()
	{
		return MAX_FLOAT;
	}
};

template<>
struct STypeLimits<double>
{
	constexpr static double Min()
	{
		return MIN_DOUBLE;
	}

	constexpr static double MAX()
	{
		return MAX_DOUBLE;
	}
};

template<typename Type>
struct SInvalidValue
{
	static Type Get()
	{
		// static_assert(false, "SInvalidValue is specialized for this type!");
	}
};

template<>
struct SInvalidValue<uint32>
{
	constexpr static uint32 Get()
	{
		return STypeLimits<uint32>::MAX();
	}
};

template<>
struct SInvalidValue<OString>
{
	constexpr static OString Get()
	{
		return "NONE";
	}
};

#define UINT32_INVALID_VALUE STypeLimits<uint32>::MAX()
#define STRING_INVALID_VALUE SInvalidValue<OString>::Get();