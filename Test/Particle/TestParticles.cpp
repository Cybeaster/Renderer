#include "TestParticles.hpp"

#include "Renderer.hpp"
#include "glm.hpp"

#ifdef USE_CUDA

// Helper functions and utilities to work with CUDA
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <helper_cuda.h>
#include <helper_functions.h>

#pragma region CUDA
__global__ void calcParticleVel(float* particlePos, float* inc, float* result, const float incMultiplier, bool UsePositiveDir)
{
	int it = blockIdx.x * blockDim.x + threadIdx.x;
	float incResult;

	if (it < 3)
	{
		if (UsePositiveDir)
			incResult = (particlePos[it] - inc[it]) * incMultiplier;
		else
			incResult = (inc[it] - particlePos[it]) * incMultiplier;
		result[it] = particlePos[it] + 1.f * (incResult - particlePos[it]);
		__syncthreads();
	}
}

/**
 * @brief Calc's particles vector speed.
 *
 * @param ParticlePos
 * @param inc
 * @param incMultiplier
 * @param UsePositiveDir Determines where particle has to move (to field or from it).
 * @return Vec3 Result - current velocity vector..
 */
Vec3 calcVelocity(const Vec3& particlePos, const Vec3& inc, float incMultiplier, bool usePositiveDir)
{
	float* host_ParticlePos = new float[3];
	host_ParticlePos[0] = particlePos.x;
	host_ParticlePos[1] = particlePos.y;
	host_ParticlePos[2] = particlePos.z;

	float* host_Inc = new float[3];

	host_Inc[0] = inc.x;
	host_Inc[1] = inc.y;
	host_Inc[2] = inc.z;

	float* host_result = new float[3];

	float* device_ParticlePos = nullptr;
	float* device_Inc = nullptr;
	float* device_Result = nullptr;

	size_t vecSize = 3 * sizeof(float);

	checkCudaErrors(cudaMalloc((void**)&device_ParticlePos, vecSize));
	checkCudaErrors(cudaMalloc((void**)&device_Inc, vecSize));
	checkCudaErrors(cudaMalloc((void**)&device_Result, vecSize));

	checkCudaErrors(cudaMemcpy(device_ParticlePos, host_ParticlePos, vecSize, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(device_Inc, host_Inc, vecSize, cudaMemcpyHostToDevice));

	calcParticleVel<<<1, 3>>>(
	    device_ParticlePos,
	    device_Inc,
	    device_Result,
	    incMultiplier,
	    usePositiveDir);

	checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(
	    cudaMemcpy(host_result, device_Result, vecSize, cudaMemcpyDeviceToHost));

	Vec3 result{ host_result[0], host_result[1], host_result[2] };

	checkCudaErrors(cudaFree(device_ParticlePos));
	checkCudaErrors(cudaFree(device_Inc));
	checkCudaErrors(cudaFree(device_Result));

	delete[] host_ParticlePos;
	delete[] host_Inc;
	delete[] host_result;

	return result;
}

__global__ void calcCudaMVMatrix(float* rotation, float* translation, float* result)
{
	int ROW = blockIdx.y * blockDim.y + threadIdx.y;
	int COL = blockIdx.x * blockDim.x + threadIdx.x;

	float tmpSum = 0;

	if (ROW < 4 && COL < 4)
	{
		// each thread computes one element of the block sub-matrix
		for (int i = 0; i < 4; i++)
		{
			tmpSum += rotation[ROW * 4 + i] * translation[i * 4 + COL];
		}
	}
	result[ROW * 4 + COL] = tmpSum;
}

/**
 * @brief Multiplies rotation by translation.
 *
 * @param rotation
 * @param translation
 * @return Mat4 Result
 */
Mat4 calcMVMatrix(const Mat4& rotation, const Mat4& translation)
{
	float host_rotation[16] = { rotation[0][0], rotation[0][1], rotation[0][2], rotation[0][3], rotation[1][0], rotation[1][1], rotation[1][2], rotation[1][3], rotation[2][0], rotation[2][1], rotation[2][2], rotation[2][3], rotation[3][0], rotation[3][1], rotation[3][2], rotation[3][3] };

	float host_translation[16] = { translation[0][0], translation[0][1], translation[0][2], translation[0][3], translation[1][0], translation[1][1], translation[1][2], translation[1][3], translation[2][0], translation[2][1], translation[2][2], translation[2][3], translation[3][0], translation[3][1], translation[3][2], translation[3][3] };

	float host_result[16];

	float* device_rotation;
	float* device_translation;
	float* device_Result;

	size_t matSize = 16 * sizeof(float);

	checkCudaErrors(cudaMalloc((void**)&device_rotation, matSize));
	checkCudaErrors(cudaMalloc((void**)&device_translation, matSize));
	checkCudaErrors(cudaMalloc((void**)&device_Result, matSize));

	checkCudaErrors(cudaMemcpy(device_rotation, host_rotation, matSize, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(device_translation, host_translation, matSize, cudaMemcpyHostToDevice));

	dim3 threadsPerBlock(4, 4);
	dim3 blocksPerGrid(1, 1);
	calcCudaMVMatrix<<<blocksPerGrid, threadsPerBlock>>>(device_rotation, device_translation, device_Result);

	checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(
	    cudaMemcpy(host_result, device_Result, matSize, cudaMemcpyDeviceToHost));

	checkCudaErrors(cudaFree(device_rotation));
	checkCudaErrors(cudaFree(device_translation));
	checkCudaErrors(cudaFree(device_Result));

	return Mat4(
	    { host_result[0], host_result[1], host_result[2], host_result[3] },
	    { host_result[4], host_result[5], host_result[6], host_result[7] },
	    { host_result[8], host_result[9], host_result[10], host_result[11] },
	    { host_result[12], host_result[13], host_result[14], host_result[15] });
}

__global__ void calcCudaTranslation(float* vMat, float* position, float* result)
{
	int xIt = blockIdx.x * blockDim.x + threadIdx.x;

	if (xIt < 4)
	{
		for (int i = 0; i < 4; i++)
		{
			if (i == 3)
				result[xIt] += vMat[i * 4 + xIt];
			else
				result[xIt] += vMat[i * 4 + xIt] * position[i];
		}
	}
}

/**
 * @brief Calcs Model-View matrix
 *
 * @param vMat Camera matrix.
 * @param position Particle position.
 * @return Mat4 Assembled Model-View matrix.
 */
Mat4 calcTranslation(const Mat4& vMat, const Vec3 position)
{
	float host_vMat[16] = { vMat[0][0], vMat[0][1], vMat[0][2], vMat[0][3], vMat[1][0], vMat[1][1], vMat[1][2], vMat[1][3], vMat[2][0], vMat[2][1], vMat[2][2], vMat[2][3], vMat[3][0], vMat[3][1], vMat[3][2], vMat[3][3] };

	float host_Position[3] = {
		position.x, position.y, position.z
	};

	float host_result[4];

	float* device_vMat;
	float* device_position;
	float* device_Result;

	size_t matSize = 16 * sizeof(float);
	size_t vecSize = 3 * sizeof(float);

	checkCudaErrors(cudaMalloc((void**)&device_vMat, matSize));
	checkCudaErrors(cudaMalloc((void**)&device_position, vecSize));
	checkCudaErrors(cudaMalloc((void**)&device_Result, vecSize));

	checkCudaErrors(cudaMemcpy(device_vMat, host_vMat, matSize, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(device_position, host_Position, vecSize, cudaMemcpyHostToDevice));

	dim3 threadsPerBlock(4, 4);
	dim3 blocksPerGrid(1, 1);

	calcCudaTranslation<<<blocksPerGrid, threadsPerBlock>>>(device_vMat, device_position, device_Result);

	checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(
	    cudaMemcpy(host_result, device_Result, vecSize, cudaMemcpyDeviceToHost));

	Mat4 result(vMat);
	result[3] = { host_result[0], host_result[1], host_result[2], 1.f };

	checkCudaErrors(cudaFree(device_vMat));
	checkCudaErrors(cudaFree(device_position));
	checkCudaErrors(cudaFree(device_Result));

	for (size_t i = 0; i < 4; i++)
		printf("%f", host_result[i]);

	return result;
}
#pragma endregion CUDA
#endif

namespace Test
{

OTestParticles::OTestParticles(OPath shaderPath, OSharedPtr<RenderAPI::ORenderer> Renderer)
    : OTest(shaderPath, Renderer)
{
	AddParticle({ -80, 20, 0 }, 1, 1, Particles45StartVel);

	auto particle = Particles[0];
	auto size = sizeof(float) * Particles[0].getVertecies().size();
	auto data = particle.getVertecies().begin()._Ptr;

	SVertexContext contextVertex(AddBuffer(data, size), 0, 3, GL_FLOAT, false, 0, 0, nullptr);

	SDrawContext drawContext(GL_TRIANGLES,
	                         0,
	                         particle.getVertecies().size() / 3,
	                         GL_LEQUAL,
	                         GL_CCW,
	                         GL_DEPTH_TEST);

	DefaultParticleHandle = CreateVertexElement(contextVertex, drawContext);

	AddField({ 80.f, -10.f, -1.f }, DefaultFieldStrenght, { 1.f, 0.f, 0.f }, 1);
	AddField({ 55.f, 20.f, -1.f }, DefaultFieldStrenght, { 1.f, 0.f, 0.f }, 1);
	AddField({ 23.f, 0.f, -1.f }, DefaultFieldStrenght, { 1.f, 0.f, 0.f }, 1);
	AddField({ -25.f, -20.f, -1.f }, DefaultFieldStrenght, { 1.f, 0.f, 0.f }, 1);
}

void OTestParticles::OnUpdate(
    const float& DeltaTime,
    const float& Aspect,
    const OVec3& CameraPos,
    OMat4& PMat,
    OMat4& VMat)
{
	OTest::OnUpdate(DeltaTime, Aspect, CameraPos, PMat, VMat);

	ParticleSpawnTick(DeltaTime);
	DrawParticles(DeltaTime, VMat);
	DrawFields(DeltaTime, VMat);
}

void OTestParticles::MoveParticle(Particle& particle, float deltaTime, OMat4 vMat)
{
	ChangeVelocity(particle);

	OMat4 translation = glm::translate(vMat, particle.getPosition());
	OMat4 rotation = particle.rotate(deltaTime);
	GetShader().SetUnformMat4f("mv_matrix", OMat4(translation * rotation));
	particle.IncreaseRotSpeed(deltaTime * 10);
	particle.move();
}

void OTestParticles::DrawParticles(float deltaTime, OMat4 vMat)
{
	for (auto& particle : Particles)
	{
		MoveParticle(particle, deltaTime, vMat);
		particle.updateColor();
		OVec3 color = particle.getColor();
		GetShader().SetUniform4f("additionalColor", color.x, color.y, color.y, 1);

		Draw(DefaultParticleHandle);
	}
}

void OTestParticles::ChangeVelocity(Particle& particle)
{
	for (auto& field : Fields) // Check distance to all fields to detect a collision.
	{
		float pointsDist = glm::distance(particle.getPosition(), field.particleField.getPosition());
		if (pointsDist < field.radius)
		{
			if (particle.getCharge() == field.particleField.getCharge()) // If the particle and field have the same charge - Move particle in opossite dir.
			{
				OVec3 inc = (field.particleField.getPosition() - particle.getPosition()) * field.fieldStrenght / pointsDist;
				OVec3 res = glm::mix(particle.getPosition(), inc, 1.f);
				particle.incVelocity(res);
			}
			else
			{
				OVec3 inc = (particle.getPosition() - field.particleField.getPosition()) * field.fieldStrenght / pointsDist;
				OVec3 res = glm::mix(particle.getPosition(), inc, 1.f);
				particle.incVelocity(res);
			}
		}
	}
}

void OTestParticles::ParticleSpawnTick(float DeltaTime)
{
	if (ParticleSpawnTimer <= 0)
	{
		ParticleSpawnTimer = ParticleSpawnTime;
		float charge = rand() % 100 > 100 ? -1 : 1;
		const OVec3 velocity = rand() % 100 > 50 ? Particles45StartVel : ParticlesNegative45StartVel;
		AddParticle({ -80, 20, 0 }, rand() % 3, charge, velocity);

		Renderer->LookAtCamera((Particles.end() - 1)->getPosition());
	}
	else
		ParticleSpawnTimer -= DeltaTime;
}

void OTestParticles::AddField(const OVec3& pos, const float& strenght, const OVec3& chargeVec, const float& charge)
{
	Particle particle(pos, {}, 76, 32, 1, -1);
	SGravityField field(35, strenght, particle, charge);
	Fields.push_back(field);
}

void OTestParticles::FieldSpawnTick(float DeltaTime)
{
	if (FieldSpawnTimer <= 0)
	{
		FieldSpawnTimer = FieldSpawnTime;
		AddField({ 25.f, -10.f, -1.f }, 10, { 1.f, 0.f, 0.f }, 1);
	}
	else
		FieldSpawnTimer -= DeltaTime;
}

void OTestParticles::AddParticle(const OVec3& startPos, const float& radius, const float& charge, const OVec3& startVelocity)
{
	OVec3 randPos{ float(rand() % 15), float(rand() % 15), 0.f };
	Particles.push_back(Particle(startPos + randPos, startVelocity, 36, 18, radius, charge));
}

void OTestParticles::DrawFields(float deltaTime, OMat4 vMat)
{
	for (auto& field : Fields)
	{
		OMat4 rot = field.particleField.rotate(deltaTime);
		OMat4 translation = glm::translate(vMat, field.particleField.getPosition());

		field.particleField.IncreaseRotSpeed(10);

		GetShader().SetUnformMat4f("mv_matrix", (translation * rot));
		GetShader().SetUniform4f("additionalColor", 1, 0, 0, 1);

		Draw(DefaultParticleHandle);
	}
}

} // namespace Test
