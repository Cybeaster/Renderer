#pragma once

#include "AllocatorUtils.hpp"
#include "Types.hpp"

#include <type_traits>

namespace RenderAPI
{
#define SMALL_INLINE_ALLOC_STACK_SIZE_MSG                             \
	"MaxStackSize is smaller or equal to the size of a pointer. '\n'" \
	"This will make the use of an InlineAllocator pointless. '\n'"    \
	"Please increase the MaxStackSize."

/// @brief Allows allocation on the stack
/// @tparam MaxStackSize Stack Size
template<size_t MaxStackSize>
class OInlineAllocator : public SInlineAllocatable
{
public:
	OInlineAllocator()
	{
		static_assert(MaxStackSize > sizeof(void*), SMALL_INLINE_ALLOC_STACK_SIZE_MSG);
	}

	~OInlineAllocator() noexcept { Free(); }

	OInlineAllocator(const OInlineAllocator& Other) noexcept
	{
		ConstructFrom(Other);
	}

	OInlineAllocator& operator=(const OInlineAllocator& Other)
	{
		ConstructFrom(Other);
		return *this;
	}

	OInlineAllocator(OInlineAllocator&& Other) noexcept
	    : AllocSize(Other.AllocSize)
	{}

	OInlineAllocator& operator=(OInlineAllocator&& Other) noexcept
	{
		Free();
		AllocSize = Other.AllocSize;
		MoveFrom(Move(Other));
		return *this;
	}

	void ConstructFrom(const OInlineAllocator& Other)
	{
		if (Other.IsAllocated())
		{
			SAllocatorUtils::MemCopy(Allocate(Other.AllocSize), Other.GetAllocation(), Other.AllocSize);
		}
		AllocSize = Other.AllocSize;
	}
	void MoveFrom(OInlineAllocator&& Other)
	{
		Other.AllocSize = 0;
		if (AllocSize > MaxStackSize)
		{
			SAllocatorUtils::Swap(Pointer, Other.Pointer);
		}
		else
		{
			SAllocatorUtils::MemCopy(Buffer, Other.Buffer, AllocSize);
		}
	}

	void* Allocate(const uint32& Size)
	{
		if (AllocSize != Size)
		{
			Free();
			AllocSize = Size;
			Pointer = SAllocatorUtils::Allocate(Size);
		}
	}

	void Free()
	{
		if (AllocSize > MaxStackSize)
		{
			SAllocatorUtils::Free(Pointer);
		}
		AllocSize = 0;
	}

	template<typename AllocationType>
	NODISCARD AllocationType* GetAllocation() const
	{
		if (IsAllocated())
		{
			return IsHeapAllocated() ? // NOLINT
			           (AllocationType*)Pointer : // NOLINT
			           (AllocationType*)Buffer; // NOLINT
		}
		return nullptr;
	}

	NODISCARD uint32 GetSize() const
	{
		return AllocSize;
	}

	NODISCARD bool IsAllocated() const
	{
		return AllocSize > 0;
	}

	NODISCARD bool IsHeapAllocated() const
	{
		return AllocSize > MaxStackSize;
	}

private:
	union
	{
		int8 Buffer[MaxStackSize];
		void* Pointer;
	};

	uint32 AllocSize{};
};

} // namespace RenderAPI
