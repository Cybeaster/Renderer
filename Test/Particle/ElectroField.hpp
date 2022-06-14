#pragma once
#include <glm.hpp>
#include "Particle.hpp"

struct GravityField
{
    float radius = 500.f;
    float fieldStrenght;

    test::Particle particleField;

    GravityField() = default;
    GravityField(const float& rad ,const float& strenght,const test::Particle& particle,const float& charge) :
    radius(rad), fieldStrenght(strenght), particleField(particle)
        {    
            particleField.setCharge(charge);
        } 
};


struct ElectroField
{
    float radius = 500.f;
    float fieldStrenght = 1.f;

    glm::vec3 powerVector{};
    glm::vec3 position{};
    ElectroField() = default;
    ElectroField(const float& rad ,const float& strenght,const float& charge,const glm::vec3& power,const glm::vec3& pos) :
    radius(rad), fieldStrenght(strenght), powerVector(power),position(pos)
        {    
        } 
};