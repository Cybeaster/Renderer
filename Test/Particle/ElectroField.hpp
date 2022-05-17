#pragma once
#include <glm.hpp>
#include "Particle.hpp"

struct ElectroField
{
    float radius = 500.f;
    float fieldStrenght;
    test::Particle particleField;

    ElectroField() = default;
    ElectroField(const float& rad ,const float& strenght,const test::Particle& particle,const float& charge) :
    radius(rad), fieldStrenght(strenght), particleField(particle)
        {    
            particleField.setCharge(charge);
        } 
};
