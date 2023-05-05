#pragma once

#include "Math.hpp"
namespace RAPI
{

struct SMaterialSpec
{
	OVec4 Ambient;
	OVec4 Diffuse;
	OVec4 Specular;
	float Shininess;
};

class OMaterial
{
	static SMaterialSpec GoldMaterial;
	static SMaterialSpec SilverMaterial;
	static SMaterialSpec BronzeMaterial;

private:
	SMaterialSpec Spec;
};
} // namespace RAPI
