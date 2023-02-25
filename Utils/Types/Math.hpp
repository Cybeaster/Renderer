#pragma once
#include "TypeTraits.hpp"

#include <glm.hpp>
#include <gtc/matrix_transform.hpp>
#include <gtc/type_ptr.hpp>

using OMat4 = glm::mat4;

using OVec2 = glm::vec2;
using OVec3 = glm::vec3;
using OVec4 = glm::vec4;

struct SMath
{
	static inline constexpr float Pi = 3.14159F;

	NODISCARD static auto ToRadians(float Degree) noexcept
	{
		return (Degree * 2.F * Pi) / 360.F;
	}
};
