#pragma once
#include "../Functions.hpp"
#include "TupleElem.hpp"

namespace RenderAPI
{
template<typename FunctionType, typename... ArgTypes>
auto CallFunction(FunctionType&& FuncType, ArgTypes&&... Args) -> decltype(std::forward<FunctionType>(FuncType)(std::forward<ArgTypes>(Args)...))
{
	return Forward(FuncType)(Forward(Args)...);
}
#pragma region TupleBase

template<typename Indices, typename... Types>
struct OTupleBase;

template<uint32... Indices, typename... Types>
struct OTupleBase<STIntegerSequenceWrapper<uint32, Indices...>, Types...> : TTupleElem<Types, Indices, sizeof...(Types)>...
{
	class TTFunctor;
	friend TTFunctor;

	template<typename... ArgTypes>
	explicit OTupleBase(ArgTypes... Args)
	    : TTupleElem<ArgTypes, Indices, sizeof...(ArgTypes)>(std::forward<ArgTypes>(Args))...
	{
	}
	~OTupleBase() = default;

	void Empty()
	{
	}
#pragma region GetByIndex
	template<uint32 Index>
	decltype(auto) Get() &&
	{
		return TTupleElemGetterByIndex<Index, sizeof...(Types)>::Get(static_cast<OTupleBase&&>(*this));
	}

	template<uint32 Index>
	decltype(auto) Get() const&
	{
		return TTupleElemGetterByIndex<Index, sizeof...(Types)>::Get(static_cast<const OTupleBase&>(*this));
	}

	template<uint32 Index>
	decltype(auto) Get() &
	{
		return TTupleElemGetterByIndex<Index, sizeof...(Types)>::Get(static_cast<OTupleBase&>(*this));
	}

	template<uint32 Index>
	decltype(auto) Get() const&&
	{
		return TTupleElemGetterByIndex<Index, sizeof...(Types)>::Get(static_cast<const OTupleBase&&>(*this));
	}
#pragma endregion GetByIndex
#pragma region GetByType
	template<typename T>
	decltype(auto) Get() &
	{
		return TTupleElemGetterByType<T, sizeof...(Types)>::Get(static_cast<OTupleBase&>(*this));
	}
	template<typename T>
	decltype(auto) Get() &&
	{
		return TTupleElemGetterByType<T, sizeof...(Types)>::Get(static_cast<OTupleBase&&>(*this));
	}
	template<typename T>
	decltype(auto) Get() const&
	{
		return TTupleElemGetterByType<T, sizeof...(Types)>::Get(static_cast<const OTupleBase&>(*this));
	}
	template<typename T>
	decltype(auto) Get() const&&
	{
		return TTupleElemGetterByType<T, sizeof...(Types)>::Get(static_cast<const OTupleBase&&>(*this));
	}
#pragma endregion GetByType

	template<typename FuncType, typename... ArgTypes>
	decltype(auto) Call(FuncType&& Function, ArgTypes&&... Args) &&
	{
		::Execute(Function, Args..., static_cast<OTupleBase&&>(*this).Get<Indices>()...);
	}

	template<typename FuncType, typename... ArgTypes>
	decltype(auto) Call(FuncType&& Function, ArgTypes&&... Args) &
	{
		::Execute(Function, Args..., static_cast<OTupleBase&>(*this).Get<Indices>()...);
	}

	template<typename FuncType, typename... ArgTypes>
	decltype(auto) Call(FuncType&& Function, ArgTypes&&... Args) const&&
	{
		::Execute(Function, Args..., static_cast<const OTupleBase&&>(*this).Get<Indices>()...);
	}

	template<typename FuncType, typename... ArgTypes>
	decltype(auto) Call(FuncType&& Function, ArgTypes&&... Args) const&
	{
		return ::Execute(Function, Args..., static_cast<const OTupleBase&>(*this).Get<Indices>()...);
	}
};

#pragma endregion TupleBase

#pragma region Tuple
template<typename... Types>
class OTElemSequenceTuple;

template<typename... Types>
class OTElemSequenceTuple : public OTupleBase<TMakeIntegerSequence<uint32, sizeof...(Types)>, Types...>
{
	using IntegerSequence = TMakeIntegerSequence<uint32, sizeof...(Types)>;
	using Super = OTupleBase<IntegerSequence, Types...>;

public:
	template<typename... OtherTypes>
	OTElemSequenceTuple& operator=(const OTElemSequenceTuple<OtherTypes...>& Tuple)
	{
		Assign(*this, Tuple, TMakeIntegerSequence<uint32, sizeof...(Types)>{});
		return *this;
	}

	template<typename... OtherTypes>
	OTElemSequenceTuple& operator=(OTElemSequenceTuple<OtherTypes...>&& Tuple)
	{
		Assign(*this, Tuple, TMakeIntegerSequence<uint32, sizeof...(Types)>{});
		return *this;
	}

	template<typename FirstTupleType, typename SecondTupleType, uint32... Indices>
	static void Assign(FirstTupleType& FirstTuple, SecondTupleType&& SecondTuple, STIntegerSequenceWrapper<uint32, Indices...>)
	{
		// This should be implemented with a fold expression when our compilers support it
		int temp[] = { 0, (FirstTuple.template Get<Indices>() = SecondTuple.template Get<Indices>(), 0)... };
		SecondTuple.Empty();
	}

	template<typename FirstTupleType, typename SecondTupleType, uint32... Indices>
	static void Assign(FirstTupleType& FirstTuple, SecondTupleType& SecondTuple, STIntegerSequenceWrapper<uint32, Indices...>)
	{
		// This should be implemented with a fold expression when our compilers support it
		int temp[] = { 0, (FirstTuple.template Get<Indices>() = std::forward<SecondTupleType>(SecondTuple).template Get<Indices>(), 0)... };
	}

	template<typename... ArgTypes>
	explicit OTElemSequenceTuple(ArgTypes... Args)
	    : Super(std::forward<ArgTypes>(Args)...)
	{
	}
	~OTElemSequenceTuple() = default;
};

#pragma endregion Tuple

} // namespace RenderAPI
