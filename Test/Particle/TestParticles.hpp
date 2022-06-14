#pragma once
#include <vector>
#include "Test.hpp"
#include "Particle.hpp"
#include "ElectroField.hpp"
namespace test
{

    /**
     * @brief Spawns particles in a specific area, with start velocity, <<field>> and additional parameters like force of gravity.
     * 
     * 
     */
    class TestParticles : public Test
    {

    private:

        void addField(const glm::vec3& pos,const float& strenght,const glm::vec3& chargeVec,const float& charge);
        void addParticle(const glm::vec3& startPos,const float& radius,const float& charge,const glm::vec3& startVelocity);
         /**
         * @brief Calculates timer for spawning particles.
         * 
         * @param DeltaTime 
         */
        void particleSpawnTick(float DeltaTime);

        /**
         * @brief Calculates time for fields spawning.
         * 
         * @param DeltaTime 
         */
        void fieldSpawnTick(float DeltaTime);

        /**
         * @brief Draws particles.
         * 
         * @param deltaTime Time between frames.
         * @param vMat Camera matrix.
         */
        void drawParticles(float deltaTime,glm::mat4 vMat);

        /**
         * @brief Draws fiels.
         * @param deltaTime Time between frames.
         * @param vMat Camera matrix.
         */
        void drawFields(float deltaTime,glm::mat4 vMat);

        /**
         * @brief Moves a particle each frame. 
         * 
         * @param particle Particle that has to be moved.
         * @param deltaTime Time between frames.
         * @param vMat Camera matrix.
         */
        void moveParticle(Particle& particle,float deltaTime, glm::mat4 vMat);


        /**
         * @brief Already spawned particles.
         * 
         */
        std::vector<Particle> particles;
         /**
         * @brief Already spawned fields.
         * 
         */
        std::vector<GravityField> electroFields;

        /**
         * @brief Timer for particles.
         * 
         */
        float particleSpawnTimer = 0.f;
        /**
         * @brief Time that is required to spawn each particle.
         * 
         */
        float particleSpawnTime = 0.05f;

        /**
         * @brief Timer for fields.
         * 
         */
        float fieldSpawnTimer = 0.f;
        /**
         * @brief Time for spawning fields
         * 
         */
        float fieldSpawnTime = 100.f;
       

        /**
         * @brief Start speed for particles.
         * 
         */
        const glm::vec3 particles45StartVel{0.95f,0.1f,0};
        const glm::vec3 particlesNegative45StartVel{0.95f,-0.5f,0};

        const float defaultFieldStrenght = 1.1f;
    public:
    
        TestParticles(std::string shaderPath);
        ~TestParticles();

        void OnUpdate(
        float deltaTime,
        float aspect,
        const glm::vec3& cameraPos,
        glm::mat4& pMat,
        glm::mat4& vMat) override;
        
    };
    
} // namespace test
