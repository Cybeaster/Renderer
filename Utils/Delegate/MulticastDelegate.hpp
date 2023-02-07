#pragma once
#include "../Types/Types.hpp"
#include "Delegate.hpp"
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

	[[nodiscard]] bool IsValid() const { return LocalID != INVALID_ID; }

	void Reset() noexcept { LocalID = INVALID_ID; }

private:
	uint32 LocalID;
	static uint32 SGlobalID;

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
class OMulticastDelegate
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
			Delegate.Execute(Forward<Args>(Args)...);
		}

		FORCEINLINE void Clear() { Delegate.Clear(); }

		FORCEINLINE bool IsBoundTo(const SDelegateHandle& Other)
		{
			return Handler == Other;
		}

		SDelegateHandlerPair()
		    : Handler(false) {}

		SDelegateHandlerPair(const SDelegateHandle& Handle, DelegateType&& Callback)
		    : Handler(Move(Handle)), Delegate(Move(Callback)) {}

		SDelegateHandlerPair(const SDelegateHandle& Handle,
		                     const DelegateType& Callback)
		    : Handler(Handle), Delegate(Callback) {}
	};

	template<typename ObjectType, typename... PayloadArgs>
	using TConstMemberFunc =
	    typename TTMemberFunctionType<ObjectType, void, ArgTypes...,
	                                  PayloadArgs...>::TConstFunction;

	template<typename ObjectType, typename... PayloadArgs>
	using TMemberFunc =
	    typename TTMemberFunctionType<ObjectType, void, ArgTypes...,
	                                  PayloadArgs...>::TFunction;

public:
	constexpr OMulticastDelegate() = default;

	~OMulticastDelegate() noexcept = default;

	OMulticastDelegate(const OMulticastDelegate& Other) = default;

	OMulticastDelegate(OMulticastDelegate&& Other) noexcept
	    : Events(Move(Other.Events)), Locks(Move(Other.Locks)) {}

	OMulticastDelegate& operator=(const OMulticastDelegate& Delegate) = default;
	OMulticastDelegate& operator=(OMulticastDelegate&& Delegate) noexcept
	{
		Events = Move(Delegate.Events);
		Locks = Move(Delegate.Locks);
	}

	template<typename ObjectType>
	bool RemoveFrom(ObjectType* Object);

	bool Remove(SDelegateHandle& Handler);
	bool IsBoundTo(SDelegateHandle& Handler);
	void RemoveAll();
	void Resize(const uint32& MaxSize = 0);
	void Broadcast(ArgTypes... Args);

	FORCEINLINE void GetSize() const { return Events.size(); }

private:
	void Lock() { ++Locks; }

	void Unlcok()
	{
		DELEGATE_ASSERT(Locks > 0);
		--Locks;
	}

	bool IsLocked() { return Locks > 0; }

	OVector<SDelegateHandlerPair> Events;
	uint32 Locks{};
};

template<typename... ArgTypes>
void OMulticastDelegate<ArgTypes...>::Resize(const uint32& MaxSize)
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
bool OMulticastDelegate<ArgTypes...>::IsBoundTo(SDelegateHandle& Handler)
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
void OMulticastDelegate<ArgTypes...>::RemoveAll()
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
bool OMulticastDelegate<ArgTypes...>::Remove(SDelegateHandle& Handler)
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
void OMulticastDelegate<ArgTypes...>::Broadcast(ArgTypes... Args)
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
bool OMulticastDelegate<ArgTypes...>::RemoveFrom(ObjectType* Object)
{
	if (Object != nullptr)
	{
		if (IsLocked())
		{
			for (const SDelegateHandlerPair& event : Events)
			{
				if (event.Delegate.GetOwner() == Object)
				{
					if (IsLocked())
					{
						event.Delegate.Clear();
					}
					else
					{
						std::swap(event);
					}
				}
			}
		}
	}
}

} // namespace RenderAPI
