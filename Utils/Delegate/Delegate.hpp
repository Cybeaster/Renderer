#pragma once
#include "../Types/MemberFunctionType.hpp"
#include "DelegateBase.hpp"
namespace RenderAPI
{

template<typename RetValType, typename... ArgTypes>
class OIDelegate
{
public:
	virtual RetValType Execute(ArgTypes&&... Args) = 0;
};

template<typename RetValueType, typename... ArgTypes>
class ODelegate : public ODelegateBase
{
private:
	template<typename ObjectType, typename... PayloadArgs>
	using TConstMemberFunc =
	    typename TTMemberFunctionType<ObjectType, RetValueType, ArgTypes..., PayloadArgs...>::TConstFunction;

	template<typename ObjectType, typename... PayloadArgs>
	using TMemberFunc =
	    typename TTMemberFunctionType<ObjectType, RetValueType, ArgTypes..., PayloadArgs...>::TFunction;

public:
	using TDelegateType = OIDelegate<RetValueType, ArgTypes...>;

	template<typename ObjectType, typename... PayloadTypes>
	DELEGATE_NO_DISCARD static ODelegate AddRaw(ObjectType* Object, TMemberFunc Function, PayloadTypes... Payload)
	{
		ODelegate delegate;
	}

private:
	template<typename ObjectType, typename... PayloadTypes>
	void Bind(PayloadTypes&&...  /*Args*/)
	{
		Release();
		void* bunch = Allocator.Allocate(sizeof(ObjectType));
		new (bunch) ObjectType(std::forward(PayloadTypes));
	}
};

} // namespace RenderAPI
