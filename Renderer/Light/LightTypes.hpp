#pragma once
#include "Math.hpp"

namespace RAPI
{

struct SLightBase
{
	OVec4 Ambient{};
	OVec4 Diffuse{};
	OVec4 Specular{};
};

struct SDirectionalLight : SLightBase
{
	OVec3 Direction{};
};

struct SPositionalLight : SLightBase
{
	OVec3 Position{};
};

struct SSpotlight : SLightBase
{
	OVec3 Direction{};
	OVec3 Position{};
	float Cutoff{ 0.0F };
};

struct SAttenuationFactor
{
	float Constant{ 0 };
	float Quadratic{ 0 };
	float Linear{ 0 };
};
} // namespace RAPI
