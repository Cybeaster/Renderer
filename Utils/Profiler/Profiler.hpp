#pragma once

#include "HashMap/Hash.hpp"
#include "TypeTraits.hpp"
namespace RenderAPI
{

// I need to have some stat group.
//  Stat-Group is a string-ID for this group
//  Group itself is an array of subgroups (possibly stat counters, or others)


struct SProfilerGroup
{
	uint32 ID;
};

class OCycleStatGroup
{

};

// Singleton profiler
class OProfiler
{

	OProfiler* Get()
	{
		if(Profiler == nullptr)
		{
			Profiler = new OProfiler;
		}
		return Profiler;
	}

private:

	OHashMap<SProfilerGroup, > ;

	OProfiler* Profiler= nullptr;
};

} // namespace RenderAPI
