#pragma once
#include <glm.hpp>


namespace test
{
    class Particle
    {
    public:

        Particle(glm::vec3 pos, glm::vec3 vel);
        ~Particle();

        inline uint32_t GetNumOfVertices()const
        {return numOfVertices;}
    private:
        /* data */

        glm::vec3 postition;
        glm::vec3 velocity;
        
        uint32_t numOfVertices = 12;
    };
    
    
    
} // namespace test
