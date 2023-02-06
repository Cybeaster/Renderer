#pragma once
#include "ElectroField.hpp"
#include "Particle.hpp"
#include "Test.hpp"

#include <vector>


namespace Test
{
/**
 * @brief Spawns particles in a specific area, with start velocity, <<field>> and additional parameters like force of gravity.
 *
 *
 */
class TestParticles : public OTest
{
public:
	TestParticles(TPath shaderPath, TTSharedPtr<RenderAPI::TRenderer> Renderer);

	void OnUpdate(
	    float deltaTime,
	    float aspect,
	    const TVec3& cameraPos,
	    TMat4& pMat,
	    TMat4& vMat) override;

private:
	TDrawVertexHandle DefaultParticleHandle;
	void ChangeVelocity(Particle& particle);
	void AddField(const TVec3& pos, const float& strenght, const TVec3& chargeVec, const float& charge);
	void AddParticle(const TVec3& startPos, const float& radius, const float& charge, const TVec3& startVelocity);
	/**
	 * @brief Calculates timer for spawning particles.
	 *
	 * @param DeltaTime
	 */
	void ParticleSpawnTick(float DeltaTime);

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
	void DrawParticles(float deltaTime, TMat4 vMat);

	/**
	 * @brief Draws fiels.
	 * @param deltaTime Time between frames.
	 * @param vMat Camera matrix.
	 */
	void DrawFields(float deltaTime, TMat4 vMat);

	/**
	 * @brief Moves a particle each frame.
	 *
	 * @param particle Particle that has to be moved.
	 * @param deltaTime Time between frames.
	 * @param vMat Camera matrix.
	 */
	void MoveParticle(Particle& particle, float deltaTime, TMat4 vMat);

	/**
	 * @brief Already spawned particles.
	 *
	 */
	TVector<Particle> Particles;
	/**
	 * @brief Already spawned fields.
	 *
	 */
	TVector<GravityField> Fields;

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
	const TVec3 Particles45StartVel{ 0.95f, 0.1f, 0 };
	const TVec3 ParticlesNegative45StartVel{ 0.95f, -0.5f, 0 };

	const float DefaultFieldStrenght = 1.1f;
};

} // namespace Test
