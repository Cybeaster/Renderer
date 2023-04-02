#pragma once

#include "Threads/Thread.hpp"
#include "TypeTraits.hpp"
namespace RenderAPI
{
class OBarrier
{
public:
	OBarrier() = default;
	explicit OBarrier(uint32 CountArg)
	    : Count(CountArg), Spaces(CountArg) {}

	void Wait()
	{
		const uint32 locGeneration = Generation.load();
		if (!(--Spaces))
		{
			Spaces = Count.load();
			++Generation;
		}
		else
		{
			while (Generation.load() == locGeneration)
			{
				NThisThread::Yield();
			}
		}
	}

	void DoneWaiting()
	{
		--Count;
		if (!(--Spaces))
		{
			Spaces = Count.load();
			++Generation;
		}
	}

private:
	OAtomic<uint32> Spaces{ 0 };
	OAtomic<uint32> Generation{ 0 };
	OAtomic<uint32> Count{ 0 };
};
} // namespace RenderAPI