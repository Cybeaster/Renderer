#pragma once
#include <vector>
#include "Test.hpp"
#include "Particle.hpp"

namespace test
{
    class TestParticles : public Test
    {

    private:
        void addParticle();
        void spawnTick(float DeltaTime);

        std::vector<Particle> particles;


        float particleSpawnTimer = 0.f;
        float particleSpawnTime = 0.1f;
       
    public:
    
        TestParticles(std::string shaderPath);
        ~TestParticles();

        void OnUpdate(GLFWwindow* window,
        float deltaTime,
        float aspect,
        const glm::vec3& cameraPos,
        glm::mat4& pMat,
        glm::mat4& vMat) override;
        
        void DrawParticle(const glm::vec3& Pos, float radius);
    };
    
} // namespace test
