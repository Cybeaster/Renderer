#pragma once
#include <vector>
#include "Test.hpp"
#include "Particle.hpp"
#include "ElectroField.hpp"
namespace test
{
    class TestParticles : public Test
    {

    private:

        void addField(const glm::vec3& pos,const float& strenght,const glm::vec3& chargeVec,const float& charge);
        void addParticle();
        void particleSpawnTick(float DeltaTime);
        void fieldSpawnTick(float DeltaTime);
        void drawParticles(float deltaTime,glm::mat4 vMat);
        void drawFields(float deltaTime,glm::mat4 vMat);
        void moveParticle(Particle& particle,float deltaTime, glm::mat4 vMat);

        std::vector<Particle> particles;
        std::vector<ElectroField> electroFields;

        float particleSpawnTimer = 0.f;
        float particleSpawnTime = 0.1f;

        float fieldSpawnTimer = 0.f;
        float fieldSpawnTime = 100.f;
       
        const glm::vec3 particlesStartVel{0.95f,0.1f,0};
    public:
    
        TestParticles(std::string shaderPath);
        ~TestParticles();

        void OnUpdate(GLFWwindow* window,
        float deltaTime,
        float aspect,
        const glm::vec3& cameraPos,
        glm::mat4& pMat,
        glm::mat4& vMat) override;
        
    };
    
} // namespace test
