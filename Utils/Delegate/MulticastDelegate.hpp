#pragma once
#include "../Types/Types.hpp"
#include "Delegate.hpp"
#include "RawDelegate.hpp"
#include "SmartPtr.hpp"
#include "Vector.hpp"

#include <vector>

namespace RenderAPI
{
#define INVALID_ID UINT32_MAX
struct SDelegateHandle
{
public:
	constexpr SDelegateHandle() noexcept
	    : LocalID(INVALID_ID) {}

	explicit SDelegateHandle(bool GenID)
	    : LocalID(GenID ? GetNewID() : INVALID_ID) {}

	SDelegateHandle& operator=(const SDelegateHandle& Other) = default;
	~SDelegateHandle() noexcept = default;

	SDelegateHandle(SDelegateHandle&& Other) noexcept
	    : LocalID(Other.LocalID)
	{
		Other.Reset();
	}

	SDelegateHandle(const SDelegateHandle& Other) = default;

	SDelegateHandle& operator=(SDelegateHandle&& Other) noexcept
	{
		LocalID = Other.LocalID;
		Other.Reset();
		return *this;
	}

	explicit operator bool() const noexcept { return IsValid(); }

	bool operator<(const SDelegateHandle& Other) const noexcept
	{
		return LocalID < Other.LocalID;
	}

	bool operator==(const SDelegateHandle& Other) const noexcept
	{
		return LocalID == Other.LocalID;
	}

	NODISCARD bool IsValid() const { return LocalID != INVALID_ID; }

	void Reset() noexcept { LocalID = INVALID_ID; }

private:
	uint32 LocalID;
	inline static uint32 SGlobalID = 0;

	static uint32 GetNewID()
	{
		uint32 result = SGlobalID++;
		if (SGlobalID == INVALID_ID)
		{
			SGlobalID = 0;
		}
		return result;
	}
};

template<typename... ArgTypes>
class OTMulticastDelegate
{
public:
	using DelegateType = ODelegate<void, ArgTypes...>;

private:
	struct SDelegateHandlerPair
	{
		SDelegateHandle Handler;
		DelegateType Delegate;

		NODISCARD FORCEINLINE bool IsValid() const { return Handler.IsValid(); }

		template<typename... Types>
		FORCEINLINE void Call(Types... Args)
		{
			Delegate.Execute(Forward<Types>(Args)...);
		}

		FORCEINLINE void Clear()
		{
			Delegate.Clear();
		}

		FORCEINLINE bool IsBoundTo(const SDelegateHandle& Other)
		{
			return Handler == Other;
		}

		SDelegateHandlerPair()
		    : Handler(false) {}

		SDelegateHandlerPair(const SDelegateHandle& Handle, DelegateType&& Callback)
		    : Handler(Move(Handle)), Delegate(Move(Callback)) {}

		SDelegateHandlerPair(const SDelegateHandle& Handle, // NOLINT
		                     const DelegateType& Callback)
		    : Handler(Handle), Delegate(Callback)
		{
		}
	};

	template<typename ObjectType, typename... PayloadArgs>
	using TConstMemberFunc =
	    typename STMemberFunctionType<true, ObjectType, void, ArgTypes..., PayloadArgs...>::Type;

	template<typename ObjectType, typename... PayloadArgs>
	using TNonConstMemberFunc =
	    typename STMemberFunctionType<false, ObjectType, void, ArgTypes..., PayloadArgs...>::Type;

public:
	constexpr OTMulticastDelegate() = default;

	~OTMulticastDelegate() noexcept = default;

	OTMulticastDelegate(const OTMulticastDelegate& Other) = default;

	OTMulticastDelegate(OTMulticastDelegate&& Other) noexcept
	    : Events(Move(Other.Events)), Locks(Move(Other.Locks)) {}

	OTMulticastDelegate& operator=(const OTMulticastDelegate& Delegate) = default;
	OTMulticastDelegate& operator=(OTMulticastDelegate&& Delegate) noexcept
	{
		Events = Move(Delegate.Events);
		Locks = Move(Delegate.Locks);
	}

	const SDelegateHandle& Add(DelegateType&& Delegate)
	{
		for (auto& delegate : Events)
		{
			if (!delegate.Handler.IsValid())
			{
				delegate = SDelegateHandlerPair(SDelegateHandle(true), Move(Delegate));
				return delegate.Handler;
			}
		}
		auto delegate = SDelegateHandlerPair(SDelegateHandle(true), Move(Delegate));
		Events.emplace_back(delegate);
		return delegate.Handler;
	}

	template<typename ObjectType, typename... Args2>
	SDelegateHandle AddRaw(ObjectType* Object, TNonConstMemberFunc<ObjectType, Args2...> Function, Args2&&... Args)
	{
		return Add(DelegateType::CreateRaw(Object, Function, Forward<Args2>(Args)...));
	}

	template<typename ObjectType, typename... Args2>
	SDelegateHandle AddRaw(ObjectType* Object, TConstMemberFunc<ObjectType, Args2...> Function, Args2&&... Args)
	{
		return Add(DelegateType::CreateRaw(Object, Function, Forward<Args2>(Args)...));
	}

	template<typename... Args2>
	SDelegateHandle AddStatic(void(Function)(ArgTypes..., Args2...), Args2&&... Args)
	{
		return Add(DelegateType::CreateStatic(Function, Forward<Args2>(Args)...));
	}

	template<typename LambdaType, typename... Args2>
	SDelegateHandle AddLambda(LambdaType&& Lambda, Args2&&... Args)
	{
		return Add(DelegateType::CreateLambda(Forward<LambdaType>(Lambda), Forward<Args2>(Args)...));
	}

	template<typename ObjectType, typename... Args2>
	SDelegateHandle AddSP(OTSharedPtr<ObjectType> Object, TNonConstMemberFunc<ObjectType, Args2...> Function, Args2&&... Args)
	{
		return Add(DelegateType::CreateSP(Object, Function, Forward<Args2>(Args)...));
	}

	template<typename ObjectType, typename... Args2>
	SDelegateHandle AddRaw(OTSharedPtr<ObjectType> Object, TConstMemberFunc<ObjectType, Args2...> Function, Args2&&... Args)
	{
		return Add(DelegateType::CreateSP(Object, Function, Forward<Args2>(Args)...));
	}

	template<typename ObjectType>
	bool RemoveFrom(ObjectType* Object);
	bool Remove(SDelegateHandle& Handler);
	bool IsBoundTo(SDelegateHandle& Handler);
	void RemoveAll();
	void Resize(const uint32& MaxSize = 0);
	void Broadcast(ArgTypes... Args);

	FORCEINLINE void GetSize() const
	{
		return Events.size();
	}

private:
	void Lock() { ++Locks; }

	void Unlock()
	{
		DELEGATE_ASSERT(Locks > 0);
		--Locks;
	}

	bool IsLocked() { return Locks > 0; }

	OTVector<SDelegateHandlerPair> Events;
	uint32 Locks{};
};

template<typename... ArgTypes>
void OTMulticastDelegate<ArgTypes...>::Resize(const uint32& MaxSize)
{
	if (!IsLocked())
	{
		uint32 toRemove = 0;
		for (uint32 it = 0; it < Events.size() - toRemove; it++)
		{
			if (!Events[it].IsValid())
			{
				std::swap(Events[it], Events[toRemove]);
				++toRemove;
			}
		}
		if (toRemove > MaxSize)
		{
			Events.resize(Events.size() - toRemove);
		}
	}
}

template<typename... ArgTypes>
bool OTMulticastDelegate<ArgTypes...>::IsBoundTo(SDelegateHandle& Handler)
{
	if (!IsLocked() && Handler.IsValid())
	{
		for (const auto& event : Events)
		{
			if (event.IsBoundTo(Handler))
			{
				return true;
			}
		}
	}
	return false;
}

template<typename... ArgTypes>
void OTMulticastDelegate<ArgTypes...>::RemoveAll()
{
	if (!IsLocked())
	{
		for (auto handler : Events)
		{
			handler.Clear();
		}
	}
	else
	{
		Events.clear();
	}
}

template<typename... ArgTypes>
bool OTMulticastDelegate<ArgTypes...>::Remove(SDelegateHandle& Handler)
{
	if (Handler.IsValid())
	{
		for (auto event : Events)
		{
			if (event.Handler == Handler)
			{
				if (IsLocked())
				{
					event.Clear();
				}
				else
				{
					std::swap(event, Events[Events.size() - 1]);
					Events.pop_back();
				}
				Handler.Reset();
				return true;
			}
		}
	}
	return false;
}

template<typename... ArgTypes>
void OTMulticastDelegate<ArgTypes...>::Broadcast(ArgTypes... Args)
{
	Lock();
	for (auto event : Events)
	{
		if (event.IsValid())
		{
			event.Call(Args...);
		}
	}
	Unlock();
}

template<typename... ArgTypes>
template<typename ObjectType>
bool OTMulticastDelegate<ArgTypes...>::RemoveFrom(ObjectType* Object)
{
	if (Object != nullptr)
	{
		if (IsLocked())
		{
			for (size_t it = 0; it < Events.size(); it++)
			{
				const auto& event = Events[it];
				auto& last = Events[Events.size() - 1];
				if (event.Delegate.GetOwner() == Object)
				{
					if (IsLocked())
					{
						event.Delegate.Clear();
					}
					else
					{
						std::swap(Events[it], last);
						Events.pop_back();
					}
				}
			}
		}
	}
}

} // namespace RenderAPI
