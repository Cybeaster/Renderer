#pragma once

#include "Thread.hpp"
#include "TypeTraits.hpp"
#include "Types.hpp"
#include "Vector.hpp"

namespace RenderAPI
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

class OThreadUtils
{
public:
	NODISCARD FORCEINLINE static OString GetFormattedThreadID();
};

} // namespace RenderAPI
