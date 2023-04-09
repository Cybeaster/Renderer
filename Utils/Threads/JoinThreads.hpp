#pragma once
#include "Threads/Thread.hpp"
#include "Vector.hpp"
namespace RAPI
{
class OJoinThreads
{
public:
	explicit OJoinThreads(OVector<OThread>& Other)
	    : Threads(Other) {}

	~OJoinThreads()
	{
		for (auto& thread : Threads)
		{
			if (thread.joinable())
			{
				thread.join();
			}
		}
	}

private:
	OVector<OThread>& Threads;
};
} // namespace RAPI