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
class OTestParticles : public OTest
{
public:
	OTestParticles(OPath shaderPath, OTSharedPtr<RenderAPI::ORenderer> Renderer);

	void OnUpdate(
	    const float& DeltaTime,
	    const float& Aspect,
	    const OVec3& CameraPos,
	    OMat4& PMat,
	    OMat4& VMat) override;

private:
	SDrawVertexHandle DefaultParticleHandle;
	void ChangeVelocity(Particle& particle);
	void AddField(const OVec3& pos, const float& strenght, const OVec3& chargeVec, const float& charge);
	void AddParticle(const OVec3& startPos, const float& radius, const float& charge, const OVec3& startVelocity);
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
	void DrawParticles(float deltaTime, OMat4 vMat);

	/**
	 * @brief Draws fiels.
	 * @param deltaTime Time between frames.
	 * @param vMat Camera matrix.
	 */
	void DrawFields(float deltaTime, OMat4 vMat);

	/**
	 * @brief Moves a particle each frame.
	 *
	 * @param particle Particle that has to be moved.
	 * @param deltaTime Time between frames.
	 * @param vMat Camera matrix.
	 */
	void MoveParticle(Particle& particle, float deltaTime, OMat4 vMat);

	/**
	 * @brief Already spawned particles.
	 *
	 */
	OTVector<Particle> Particles;
	/**
	 * @brief Already spawned fields.
	 *
	 */
	OTVector<SGravityField> Fields;

	/**
	 * @brief Timer for particles.
	 *
	 */
	float ParticleSpawnTimer = 0.F;
	/**
	 * @brief Time that is required to spawn each particle.
	 *
	 */
	float ParticleSpawnTime = 0.05F;

	/**
	 * @brief Timer for fields.
	 *
	 */
	float FieldSpawnTimer = 0.F;
	/**
	 * @brief Time for spawning fields
	 *
	 */
	float FieldSpawnTime = 100.F;

	/**
	 * @brief Start speed for particles.
	 *
	 */
	const OVec3 Particles45StartVel{ 0.95F, 0.1F, 0 };
	const OVec3 ParticlesNegative45StartVel{ 0.95F, -0.5F, 0 };

	const float DefaultFieldStrenght = 1.1F;
};

} // namespace Test
