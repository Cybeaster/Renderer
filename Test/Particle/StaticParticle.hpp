#pragma once
#include "Particle.hpp"

namespace test
{
    class StaticParticle : public Particle
    {
        public:
        StaticParticle(const glm::vec3& pos);
    };
    
    StaticParticle::StaticParticle(const glm::vec3& pos) : Particle(pos,{},12,18,1,0.5)
    {}
    
        
} // namespace test
