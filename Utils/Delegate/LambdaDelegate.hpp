#pragma once

#include "DelegateBase.hpp"
#include "Tuple.hpp"
#include "TypeTraits.hpp"

#include <type_traits>
#include <vcruntime.h>

namespace RenderAPI
{

template<typename LamdaType, typename RetType, typename... Args>
class OTLambdaDelegate
{
};

template<typename LamdaType, typename RetType, typename... Args, typename... Args2>
class OTLambdaDelegate<LamdaType, RetType(Args...), Args2...> : public OIDelegate<RetType, Args...>
{
public:
	explicit OTLambdaDelegate(const LamdaType& LamdaArg, const Args2&... Arguments) noexcept
	    : Lamda(LamdaArg), Payload(Arguments...)
	{
	}

	explicit OTLambdaDelegate(LamdaType&& LamdaArg, Args2&&... Arguments) noexcept
	    : Lamda(Forward(LamdaArg)), Payload(Forward(Arguments)...)
	{
	}

	void CloneTo(void* Destination) override
	{
		new (Destination) OTLambdaDelegate(Lamda, Payload);
	}

	RetType Execute(Args&&... Arguments)
	{
		return ExecuteImpl(Forward(Arguments)..., TMakeIndexSequenceFor<Args2...>());
	}

private:
	template<size_t... Indices>
	RetType ExecuteImpl(Args&&... Arguments, TIndexSequenceWrapper<Indices...>)
	{
		return (RetType)((Lamda)(Forward(Arguments)..., Payload.template Get<Indices>()...));
	}

	OTuple<Args2...> Payload;
	LamdaType Lamda;
};

} // namespace RenderAPI
