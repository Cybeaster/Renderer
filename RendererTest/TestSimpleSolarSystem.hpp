#pragma once
#include "Path.hpp"
#include "Test.hpp"

#include <cstdint>
#include <glm.hpp>
#include <vector>
namespace RAPI
{
class ORenderer;
/**
 * @brief Spawns 2 figures with specific offset.
 *
 */
class OTestSimpleSolarSystem : public OTest
{
public:
	~OTestSimpleSolarSystem() override = default;
	OTestSimpleSolarSystem() = default;
	OTestSimpleSolarSystem(OPath shaderPath, ORenderer* Renderer);

	void OnUpdate(
	    const float& DeltaTime,
	    const float& Aspect,
	    const OVec3& CameraPos,
	    OMat4& PMat,
	    OMat4& VMat) override;

private:
	SDrawVertexHandle cubeHandle;
	float cubePositions[108] = {
		-1.0F, 1.0F, -1.0F, -1.0F, -1.0F, -1.0F, 1.0F, -1.0F, -1.0F, 1.0F, -1.0F, -1.0F, 1.0F, 1.0F, -1.0F, -1.0F, 1.0F, -1.0F, 1.0F, -1.0F, -1.0F, 1.0F, -1.0F, 1.0F, 1.0F, 1.0F, -1.0F, 1.0F, -1.0F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F, -1.0F, 1.0F, -1.0F, 1.0F, -1.0F, -1.0F, 1.0F, 1.0F, 1.0F, 1.0F, -1.0F, -1.0F, 1.0F, -1.0F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F, -1.0F, -1.0F, 1.0F, -1.0F, -1.0F, -1.0F, -1.0F, 1.0F, 1.0F, -1.0F, -1.0F, -1.0F, -1.0F, 1.0F, -1.0F, -1.0F, 1.0F, 1.0F, -1.0F, -1.0F, 1.0F, 1.0F, -1.0F, 1.0F, 1.0F, -1.0F, -1.0F, 1.0F, -1.0F, -1.0F, -1.0F, -1.0F, -1.0F, -1.0F, -1.0F, 1.0F, -1.0F, 1.0F, -1.0F, 1.0F, 1.0F, -1.0F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F, -1.0F, 1.0F, 1.0F, -1.0F, 1.0F, -1.0F
	};

	SDrawVertexHandle pyramidHandle;
	float pyramidPositions[54] = {
		-1.0F, -1.0F, 1.0F, 1.0F, -1.0F, 1.0F, 0.0F, 1.0F, 0.0F, // front face
		1.0F,
		-1.0F,
		1.0F,
		1.0F,
		-1.0F,
		-1.0F,
		0.0F,
		1.0F,
		0.0F, // right face
		1.0F,
		-1.0F,
		-1.0F,
		-1.0F,
		-1.0F,
		-1.0F,
		0.0F,
		1.0F,
		0.0F, // back face
		-1.0F,
		-1.0F,
		-1.0F,
		-1.0F,
		-1.0F,
		1.0F,
		0.0F,
		1.0F,
		0.0F, // left face
		-1.0F,
		-1.0F,
		-1.0F,
		1.0F,
		-1.0F,
		1.0F,
		-1.0F,
		-1.0F,
		1.0F, // base – left front
		1.0F,
		-1.0F,
		1.0F,
		-1.0F,
		-1.0F,
		-1.0F,
		1.0F,
		-1.0F,
		-1.0F // base – right back
	};
};

} // namespace RAPI
