#pragma once

#ifndef RENDERAPI_HAZARDPOINTER_HPP
#define RENDERAPI_HAZARDPOINTER_HPP

#include "Assert.hpp"
#include "TypeTraits.hpp"
#include "Types.hpp"
#include "Utils/Types/Threads/Thread.hpp"
namespace RenderAPI
{

struct SHazardPointer
{
	OAtomic<ThreadID> ID;
	OAtomic<void*> Pointer;
};

template<uint32 Size>
class OHazardPointerManagerImpl
{
public:
	static inline SHazardPointer Pointers[Size];
	NODISCARD static constexpr uint32 GetSize()
	{
		return Size;
	}

	static OAtomic<void*>& GetPointerForCurrentThread();
	static bool IsOutstandingHazardPointerFor(void* Pointer);
};

using OHazardPointerManager = OHazardPointerManagerImpl<100>;
class OHazardPointerOwner
{
public:
	REMOVE_COPY_FOR(OHazardPointerOwner)
	OHazardPointerOwner()
	{
		for (uint32 i = 0; i < OHazardPointerManager::GetSize(); ++i)
		{
			ThreadID oldID;
			if (OHazardPointerManager::Pointers[i].ID.compare_exchange_strong(oldID, GetThisThreadID()))
			{
				Pointer = &OHazardPointerManager::Pointers[i];
				break;
			}
		}
		ASSERT_MSG(Pointer, "No hazard pointers available!")
	}

	~OHazardPointerOwner()
	{
		Pointer->Pointer.store(nullptr);
		Pointer->ID.store(ThreadID());
	}

	OAtomic<void*>& GetPointer()
	{
		Pointer->Pointer;
	}

private:
	SHazardPointer* Pointer{ nullptr };
};

template<uint32 Size>
OAtomic<void*>& OHazardPointerManagerImpl<Size>::GetPointerForCurrentThread()
{
	thread_local static OHazardPointerOwner owner;
	return owner.GetPointer();
}

template<uint32 Size>
bool OHazardPointerManagerImpl<Size>::IsOutstandingHazardPointerFor(void* Pointer)
{
	for (auto& pointer : Pointers)
	{
		if (pointer.Pointer.load() == Pointer)
		{
			return true;
		}
	}
	return false;
}

} // namespace RenderAPI

#endif // RENDERAPI_HAZARDPOINTER_HPP
