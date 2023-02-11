#pragma once
#include "../Types/MemberFunctionType.hpp"
#include "DelegateBase.hpp"
#include "Types.hpp"
namespace RenderAPI
{

template<typename RetValueType, typename... ArgTypes>
class ODelegate : public ODelegateBase
{
private:
	template<typename ObjectType, typename... PayloadArgs>
	using ConstMemberFunc =
	    typename STMemberFunctionType<ObjectType, RetValueType, ArgTypes..., PayloadArgs...>::TConstFunction;

	template<typename ObjectType, typename... PayloadArgs>
	using MemberFunc =
	    typename STMemberFunctionType<ObjectType, RetValueType, ArgTypes..., PayloadArgs...>::TFunction;

public:
	using TDelegateType = OIDelegate<RetValueType, ArgTypes...>;

	template<typename ObjectType, typename... PayloadTypes>
	DELEGATE_NO_DISCARD static ODelegate AddRaw(ObjectType* Object, MemberFunc<ObjectType, PayloadTypes...> Function, PayloadTypes... Payload)
	{
		ODelegate delegate;
	}

	void Execute(ArgTypes&&... Args)
	{
		DELEGATE_ASSERT(Allocator.IsAllocated());
		(static_cast<TDelegateType*>(GetDelegate()))->Execute(Forward<ArgTypes>(Args)...);
	}

private:
	template<typename ObjectType, typename... PayloadTypes>
	void Bind(PayloadTypes&&... Args)
	{
		Release();
		void* bunch = Allocator.Allocate(sizeof(ObjectType));
		new (bunch) ObjectType(std::forward(Args...));
	}
};

} // namespace RenderAPI
