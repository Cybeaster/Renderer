#include "StatGroup.hpp"

#include "Profiler.hpp"

namespace RAPI
{

OStatGroup::OStatGroup(OString&& GroupName)
{
	OProfiler::Get()->AddStatGroup(Move(GroupName), this);
}

} // namespace RAPI
