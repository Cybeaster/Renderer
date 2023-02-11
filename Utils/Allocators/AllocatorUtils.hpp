#pragma once
#include <Types.hpp>
#include <cstdlib>
#include <cstring>
#include <utility>

namespace RenderAPI
{
struct SAllocatorUtils
{
	static void Free(void* Ptr)
	{
		free(Ptr);
	}

	static void* Allocate(uint32 Size)
	{
		return malloc(Size);
	}

	static void Swap(void* FPtr, void* SPtr)
	{
		std::swap(FPtr, SPtr);
	}
	
	static void MemCopy(void* FBuffer, void const* SBuffer, uint32 Size)
	{
		std::memcpy(FBuffer, SBuffer, Size);
	}
};

struct SInlineAllocatable
{
	enum StackSize : uint8
	{
		_16 = 16,
		_32 = 32,
		_64 = 64
	};
};

} // namespace RenderAPI
