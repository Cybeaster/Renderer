#pragma once
#include "Particle.hpp"

namespace test
{
    class StaticParticle : public Particle
    {
        public:
        StaticParticle(const glm::vec3& pos);
    };
    
    StaticParticle::StaticParticle(const glm::vec3& pos) : Particle(pos,{},12,18,0.5,0)
    {}
    
        
} // namespace test
