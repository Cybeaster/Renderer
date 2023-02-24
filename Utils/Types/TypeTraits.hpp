#pragma once

#include "TypeTraits.hpp"

#include <algorithm>
#include <stdint.h>
#include <type_traits>

#define NODISCARD [[nodiscard]]

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

template<bool Flag, typename Arg = void>
struct STEnableIf
{
};

template<typename Arg>
struct STEnableIf<true, Arg>
{
};

#pragma region RemoveRef

template<typename T>
struct STRemoveRef
{
	using Type = T;
	using ConstType = const T;
};

template<typename T>
struct STRemoveRef<T&>
{
	using Type = T;
	using ConstType = const T;
};

template<typename T>
struct STRemoveRef<T&&>
{
	using Type = T;
	using ConstType = const T;
};

template<typename T>
using TRemoveRef = typename STRemoveRef<T>::Type;

#pragma endregion RemoveRef

#pragma region RemoveCV

template<typename T>
struct STRemoveCV
{
	using Type = T;
};

template<typename T>
struct STRemoveCV<const T>
{
	using Type = T;
};

template<typename Type>
using TRemoveCV = typename STRemoveCV<Type>::Type;

#pragma endregion RemoveCV

template<typename Type, Type RValue>
struct STIntegralConstant
{
	static constexpr Type Value = RValue;
	using ValueType = Type;
	using IntegralType = STIntegralConstant;

	explicit constexpr operator ValueType() const noexcept
	{
		return Value;
	}

	NODISCARD constexpr ValueType operator()() const noexcept
	{
		return Value;
	}
};

template<bool Value>
using TBoolConstant = STIntegralConstant<bool, Value>;

using TFalse = TBoolConstant<false>;

template<typename First, typename Second>
struct STIsSame : TBoolConstant<__is_same(First, Second)>
{
};

template<bool Value, typename First, typename... Remaining>
struct STDisjunction
{
	using Type = First;
};

template<class False, typename Next, typename... Rest>
struct STDisjunction<false, False, Next, Rest...>
{
	using Type = typename STDisjunction<Next::Value, Next, Rest...>::Type;
};

template<typename... Traits>
struct STDisjunctionValue : TFalse
{
};

template<typename First, typename... Rest>
struct STDisjunctionValue<First, Rest...> : STDisjunction<First::Value, First, Rest...>::Type
{
};

template<typename... Traits>
constexpr bool TDisjunction = STDisjunctionValue<Traits...>::Value;

template<typename SearchedType, typename... Types>
constexpr bool TIsAnyOf = TDisjunction<STIsSame<SearchedType, Types>...>;

template<typename Type>
constexpr bool TIsIntegral = TIsAnyOf<TRemoveCV<Type>,
                                      bool, char, signed char, unsigned char, wchar, char8,
                                      char16, char32, int16, uint16, int32, uint32,
                                      long, unsigned long, long long, unsigned long long>; // NOLINT

template<typename T, T... Indices>
struct STIntegerSequenceWrapper
{
	static_assert(TIsIntegral<T>, "Type must be integral!");

	using ValueType = T;

	NODISCARD static constexpr size_t Size() noexcept
	{
		return sizeof...(Indices);
	}
};

template<size_t... Indices>
using TIndexSequenceWrapper = STIntegerSequenceWrapper<size_t, Indices...>;

template<typename T, T Size>
using TMakeIntegerSequence = __make_integer_seq<STIntegerSequenceWrapper, T, Size>;

template<size_t Size>
using TMakeIndexSequence = TMakeIntegerSequence<size_t, Size>;

template<typename... Types>
using TMakeIndexSequenceFor = TMakeIndexSequence<sizeof...(Types)>;
