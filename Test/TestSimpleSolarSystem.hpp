#pragma once
#include <Renderer.hpp>
#include <cstdint>
#include <glm.hpp>
#include <vector>
namespace Test
{

/**
 * @brief Spawns 2 figures with specific offset.
 *
 */
class OTestSimpleSolarSystem : public OTest
{
public:
	OTestSimpleSolarSystem() = default;
	OTestSimpleSolarSystem(TPath shaderPath, OSharedPtr<RenderAPI::ORenderer> Renderer);

	void OnUpdate(
	    float deltaTime,
	    float aspect,
	    const TVec3& cameraPos,
	    TMat4& pMat,
	    TMat4& vMat) override;

private:
	TDrawVertexHandle cubeHandle;
	float cubePositions[108] = {
		-1.0F, 1.0F, -1.0F, -1.0F, -1.0F, -1.0F, 1.0F, -1.0F, -1.0F, 1.0F, -1.0F, -1.0F, 1.0F, 1.0F, -1.0F, -1.0F, 1.0F, -1.0F, 1.0F, -1.0F, -1.0F, 1.0F, -1.0F, 1.0F, 1.0F, 1.0F, -1.0F, 1.0F, -1.0F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F, -1.0F, 1.0F, -1.0F, 1.0F, -1.0F, -1.0F, 1.0F, 1.0F, 1.0F, 1.0F, -1.0F, -1.0F, 1.0F, -1.0F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F, -1.0F, -1.0F, 1.0F, -1.0F, -1.0F, -1.0F, -1.0F, 1.0F, 1.0F, -1.0F, -1.0F, -1.0F, -1.0F, 1.0F, -1.0F, -1.0F, 1.0F, 1.0F, -1.0F, -1.0F, 1.0F, 1.0F, -1.0F, 1.0F, 1.0F, -1.0F, -1.0F, 1.0F, -1.0F, -1.0F, -1.0F, -1.0F, -1.0F, -1.0F, -1.0F, 1.0F, -1.0F, 1.0F, -1.0F, 1.0F, 1.0F, -1.0F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F, -1.0F, 1.0F, 1.0F, -1.0F, 1.0F, -1.0F
	};

	TDrawVertexHandle pyramidHandle;
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

} // namespace Test
