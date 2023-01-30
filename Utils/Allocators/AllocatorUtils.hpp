#pragma once
#include <cstdlib>
#include <utility>
#include <cstring>
#include <Types.hpp>

namespace RenderAPI
{
    class AllocatorUtils
    {
        static void Free(void *Ptr)
        {
            free(Ptr);
        }

        static void *Allocate(uint32 Size)
        {
            return malloc(Size);
        }

        static void Swap(void *FPtr, void *SPtr)
        {
            std::swap(FPtr, SPtr);
        }

        static void MemCopy(uint8 FBuffer, uint8 SBuffer, uint32 Size)
        {
            std::memcpy(FBuffer, SBuffer, Size);
        }
    };

    struct FInlineAllocatable
    {
        enum StackSize : uint8
        {
            _16 = 16,
            _32 = 32,
            _64 = 64
        };
    };

} // namespace RenderAPI
