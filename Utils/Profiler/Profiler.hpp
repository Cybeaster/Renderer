#pragma once

#include "HashMap/Hash.hpp"
#include "HashMap/ThreadSafeHashMap.hpp"
#include "StatGroup.hpp"
#include "TypeTraits.hpp"

namespace RAPI
{

class OProfiler
{
public:
	static OProfiler* Get()
	{
		if (Profiler == nullptr)
		{
			Profiler = new OProfiler();
		}

		return Profiler;
	}

	void AddStatGroup(const OString& Name, OStatGroup* Group);

private:
	OProfiler() = default;
	OThreadSafeHashMap<SStatGroupID<OString>, OStatGroup*, STStatGroupHash<OString>> StatGroups;
	static inline OProfiler* Profiler{ nullptr };
};

#define MAKE_STAT_GROUP(Name)

} // namespace RAPI
