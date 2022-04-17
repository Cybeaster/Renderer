#pragma once
#include <vector>
#include "Test.hpp"
#include "Particle.hpp"
namespace test
{
    class TestParticles : public Test
    {

    private:

       
    public:
        TestParticles(/* args */);
        ~TestParticles();

        void OnUpdate(GLFWwindow* window, float deltaTime);
        void DrawParticle(const glm::vec3& Pos, float radius);
    };
    
    std::vector<Particle> particles;
    
    float particleSpawnTimer = 0.f;
    float particleSpawnTime = 2.f;
} // namespace test
