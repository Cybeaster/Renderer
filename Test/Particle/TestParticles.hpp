#pragma once
#include <vector>
#include "Test.hpp"
#include "Particle.hpp"
#include "ElectroField.hpp"

namespace Test
{
    /**
     * @brief Spawns particles in a specific area, with start velocity, <<field>> and additional parameters like force of gravity.
     *
     *
     */
    class TestParticles : public Test
    {
    public:
        TestParticles(String shaderPath);

        void OnUpdate(
            float deltaTime,
            float aspect,
            const Vec3 &cameraPos,
            Mat4 &pMat,
            Mat4 &vMat) override;

    private:
        void ChangeVelocity(Particle &particle);
        void AddField(const Vec3 &pos, const float &strenght, const Vec3 &chargeVec, const float &charge);
        void AddParticle(const Vec3 &startPos, const float &radius, const float &charge, const Vec3 &startVelocity);
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
        void FieldSpawnTick(float DeltaTime);

        /**
         * @brief Draws particles.
         *
         * @param deltaTime Time between frames.
         * @param vMat Camera matrix.
         */
        void drawParticles(float deltaTime, Mat4 vMat);

        /**
         * @brief Draws fiels.
         * @param deltaTime Time between frames.
         * @param vMat Camera matrix.
         */
        void DrawFields(float deltaTime, Mat4 vMat);

        /**
         * @brief Moves a particle each frame.
         *
         * @param particle Particle that has to be moved.
         * @param deltaTime Time between frames.
         * @param vMat Camera matrix.
         */
        void MoveParticle(Particle &particle, float deltaTime, Mat4 vMat);

        /**
         * @brief Already spawned particles.
         *
         */
        Vector<Particle> Particles;
        /**
         * @brief Already spawned fields.
         *
         */
        Vector<GravityField> Fields;

        /**
         * @brief Timer for particles.
         *
         */
        float ParticleSpawnTimer = 0.f;
        /**
         * @brief Time that is required to spawn each particle.
         *
         */
        float ParticleSpawnTime = 0.05f;

        /**
         * @brief Timer for fields.
         *
         */
        float FieldSpawnTimer = 0.f;
        /**
         * @brief Time for spawning fields
         *
         */
        float FieldSpawnTime = 100.f;

        /**
         * @brief Start speed for particles.
         *
         */
        const Vec3 Particles45StartVel{0.95f, 0.1f, 0};
        const Vec3 ParticlesNegative45StartVel{0.95f, -0.5f, 0};

        const float DefaultFieldStrenght = 1.1f;
    };

} // namespace test
