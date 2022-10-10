#pragma once
#include <glm.hpp>
#include "Particle.hpp"

struct GravityField
{
    float radius = 500.f;
    float fieldStrenght;

    Test::Particle particleField;

    GravityField() = default;
    GravityField(const float &rad, const float &strenght, const Test::Particle &particle, const float &charge) : radius(rad), fieldStrenght(strenght), particleField(particle)
    {
        particleField.setCharge(charge);
    }
};
