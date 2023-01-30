#pragma once

#include "AllocatorUtils.hpp"
namespace RenderAPI
{
#define SMALL_INLINE_ALLOC_STACK_SIZE_MSG "MaxStackSize is smaller or equal to the size of a pointer. '\n'" \
                                          "This will make the use of an InlineAllocator pointless. '\n'"    \
                                          "Please increase the MaxStackSize."

    /// @brief Allows allocation on the stack
    /// @tparam MaxStackSize Stack Size
    template <size_t MaxStackSize>
    class TInlineAllocator : public FInlineAllocatable
    {
    public:
        TInlineAllocator() : AllocSize(0)
        {
            static_assert(MaxStackSize > sizeof(void *), SMALL_INLINE_ALLOC_STACK_SIZE);
        }
        ~TInlineAllocator() noexcept
        {
            Free();
        }

        TInlineAllocator(const TInlineAllocator &Other) noexcept
            : AllocSize(0)
        {
            ConstructFrom(Other);
        }

        TInlineAllocator &operator=(const TInlineAllocator &Other)
        {
            ConstructFrom(Other);
            return &this;
        }

        TInlineAllocator(TInlineAllocator &&Other) noexcept
            : AllocSize(Other.AllocSize)
        {
            Move(Other);
        }

        TInlineAllocator &operator=(TInlineAllocator &&Other) noexcept
        {
            Free();
            AllocSize = Other.AllocSize;
            MoveFrom(Other);
            return *this;
        }

        void ConstructFrom(const TInlineAllocator &Other)
        {
            if (Other.IsAlocated())
            {
                AllocatorUtils::MemCopy(Allocate(Other.AllocSize), Other.GetAllocation(), Other.AllocSize);
            }
            AllocSize = Other.AllocSize;
        }
        void MoveFrom(TInlineAllocator &&Other)
        {
            Other.AllocSize = 0;
            if (AllocSize > MaxStackSize)
            {
                AllocatorUtils::Swap(Pointer, Other.Pointer);
            }
            else
            {
                AllocatorUtils::MemCopy(Buffer, Other.Buffer, AllocSize);
            }
        }

        void *Allocate(const uint32 Size)
        {
            if (AllocSize != Size)
            {
                Free();
                AllocSize = Size;
                Pointer = AllocatorUtils::Allocate(Size);
            }
        }

        void Free()
        {
            if (AllocSize > MaxStackSize)
            {
                AllocatorUtils::Free(Pointer);
            }
            AllocSize = 0;
        }

        void *GetAllocation() const
        {
            if (HasAllocation())
            {
                return HasHeapAllocation() ? Pointer : (void *)Buffer;
            }
            else
            {
                return nullptr;
            }
        }

        uint32 GetSize() const
        {
            return AllocSize;
        }

        bool IsAlocated() const
        {
            return AllocSize > 0;
        }

        bool IsHeapAllocated() const
        {
            AllocSize > MaxStackSize;
        }

    private:
        union
        {
            int8 Buffer[MaxStackSize];
            void *Pointer;
        };

        uint32 AllocSize;
    };

    template <size_t MaxStackSize>
    TInlineAllocator<MaxStackSize>::TInlineAllocator()
    {
    }

    TInlineAllocator<MaxStackSize>::~TInlineAllocator()
    {
    }

} // namespace RenderAPI
