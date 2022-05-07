#pragma once
#include <glm.hpp>
#include "Particle.hpp"

struct ElectroField
{
    float radius = 100.f;
    float fieldStrenght;
    test::Particle particleField;

    ElectroField() = default;
    ElectroField(const float& rad ,const float& charge,const test::Particle& particle) : 
        radius(rad),fieldStrenght(charge), particleField(particle){} 
};
