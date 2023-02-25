#pragma once
#include "../Allocators/InlineAllocator.hpp"
#include "Assert.hpp"
#include "Types.hpp"

#pragma optimize("", off)
namespace RenderAPI
{

class OIDelegateBase
{
public:
	OIDelegateBase() = default;
	virtual ~OIDelegateBase() noexcept = default;
	NODISCARD virtual const void* GetOwner() const
	{
		return nullptr;
	}
	virtual void CopyTo(void* Destination) = 0;
};

template<typename RetValueType, typename... ArgTypes>
class OIDelegate : public OIDelegateBase
{
public:
	virtual RetValueType Execute(ArgTypes&&... Args) = 0;
};

class ODelegateBase
{
public:
	ODelegateBase() = default;

	ODelegateBase(ODelegateBase&& Other) noexcept
	    : Allocator(Move(Other.Allocator))
	{
	}

	virtual ~ODelegateBase() noexcept
	{
		Release();
	}
	ODelegateBase(const ODelegateBase& Other)
	{
		Copy(Other);
	}

	ODelegateBase& operator=(const ODelegateBase& Other)
	{
		Release();
		Copy(Other);
		return *this;
	}

	template<typename T>
	ODelegateBase& operator=(T&& Other) noexcept
	{
		Release();
		MoveDelegate(Move(Other));
		return *this;
	}

	template<typename OwnerType>
	NODISCARD const OwnerType* GetOwner() const
	{
		if (Allocator.IsAllocated())
		{
			return GetDelegate()->GetOwner();
		}
		return nullptr;
	}

	bool IsBoundTo(void* Object) const
	{
		if (Object == nullptr || !Allocator.IsAllocated())
		{
			return false;
		}

		return GetDelegate()->GetOwner() == Object;
	}

	NODISCARD bool IsBound() const
	{
		return Allocator.IsAllocated();
	}

	void Clear()
	{
		Release();
	}

	NODISCARD uint32 GetSize() const
	{
		return Allocator.GetSize();
	}

	void ClearIfBoundTo(void* Object)
	{
		if (Object != nullptr && IsBoundTo(Object))
		{
			Clear();
		}
	}

protected:
	FORCEINLINE void Release()
	{
		if (Allocator.IsAllocated())
		{
			auto* delegate = GetDelegate();
			if (ENSURE(delegate != nullptr))
			{
				delegate->~OIDelegateBase();
				Allocator.Free();
			}
		}
	}

	void Copy(const ODelegateBase& Other)
	{
		if (Other.Allocator.IsAllocated())
		{
			auto* delegate = Other.GetDelegate();
			Allocator.Allocate(Other.Allocator.GetSize());
			assert(delegate != nullptr);
			if (ENSURE(delegate != nullptr))
			{
				Other.GetDelegate()->CopyTo(Allocator.GetAllocation());
			}
		}
	}

	FORCEINLINE void MoveDelegate(ODelegateBase&& Other)
	{
		Allocator = Move(Other.Allocator);
	}

	NODISCARD OIDelegateBase* GetDelegate() const
	{
		return static_cast<OIDelegateBase*>(Allocator.GetAllocation());
	}

	OInlineAllocator<SInlineAllocatable::StackSize::_32> Allocator;
};
} // namespace RenderAPI

#pragma optimize("", on)