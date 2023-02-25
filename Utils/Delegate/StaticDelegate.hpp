#pragma once

#include "DelegateBase.hpp"
#include "Tuple.hpp"
#include "TypeTraits.hpp"
namespace RenderAPI
{

template<typename RetType, typename... Args2>
class OTStaticDelegate
{
};

template<typename RetType, typename... Args, typename... Args2>

class OTStaticDelegate<RetType(Args...), Args2...> : public OIDelegate<RetType, Args...>
{
public:
	using TDelegateFunction = RetType (*)(Args..., Args2...);

	OTStaticDelegate(TDelegateFunction FunctionArg, Args2&&... PayloadArg) noexcept
	    : Function(FunctionArg), Payload(Forward<Args2>(PayloadArg)...)
	{
	}

	OTStaticDelegate(TDelegateFunction FunctionArg, OTuple<Args2...> PayloadArg) noexcept
	    : Function(FunctionArg), Payload(PayloadArg)
	{
	}

	void CopyTo(void* Destination) override
	{
		new (Destination) OTStaticDelegate(Function, Payload);
	}

	RetType Execute(Args&&... Arguments) override
	{
		return ExecuteImpl(Forward<Args2>(Arguments)..., TMakeIndexSequence<Args2...>());
	}

private:
	template<size_t... Indices>
	RetType ExecuteImpl(Args&&... Arguments, TIndexSequenceWrapper<Indices...>)
	{
		return Function(Forward<Args>(Arguments)..., (Payload.template Get<Indices>())...);
	}

	TDelegateFunction Function;
	OTuple<Args2...> Payload;
};
} // namespace RenderAPI