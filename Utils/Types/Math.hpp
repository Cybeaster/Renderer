#pragma once
#include "TypeTraits.hpp"

#include <glm.hpp>
#include <gtc/matrix_transform.hpp>
#include <gtc/type_ptr.hpp>

using OMat4 = glm::mat4;

using OVec2 = glm::vec2;
using OVec3 = glm::vec3;
using OVec4 = glm::vec4;

namespace RAPI
{
struct SMath
{
	static inline constexpr float Pi = 3.14159F;

	NODISC_FORCEINL static auto ToRadians(float Degree) noexcept
	{
		return (Degree * 2.F * Pi) / 360.F;
	}

	NODISC_FORCEINL static auto Min(auto First, auto Second)
	{
		return First > Second ? Second : First;
	}

	NODISC_FORCEINL static auto Max(auto First, auto Second)
	{
		return First > Second ? First : Second;
	}
};

} // namespace RAPI
