#pragma once
#include "Particle.hpp"

#include <glm.hpp>


struct SGravityField
{
	float radius = 500.f;
	float fieldStrenght;

	Test::Particle particleField;

	SGravityField() = default;
    
	SGravityField(const float& rad, const float& strenght, const Test::Particle& particle, const float& charge)
	    : radius(rad), fieldStrenght(strenght), particleField(particle)
	{
		particleField.SetCharge(charge);
	}
};
