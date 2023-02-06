#pragma once
#include "../Renderer/Vertex/SimpleVertexHandle.hpp"

#include <Test.hpp>
#include <Texture.hpp>

class GLFWwindow;
namespace Test
{
class TestTexture : public OTest
{
public:
	TestTexture(const TPath& filePath, const TPath& ShaderPath, TTSharedPtr<RenderAPI::TRenderer> Renderer);
	~TestTexture();

	virtual void OnUpdate(
	    const float deltaTime,
	    const float aspect,
	    const TVec3& cameraPos,
	    TMat4& pMat,
	    TMat4& vMat) override;

private:
	float pyramidPositions[54] = {
		-1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 0.0f, 1.0f, 0.0f, // front face
		1.0f,
		-1.0f,
		1.0f,
		1.0f,
		-1.0f,
		-1.0f,
		0.0f,
		1.0f,
		0.0f, // right face
		1.0f,
		-1.0f,
		-1.0f,
		-1.0f,
		-1.0f,
		-1.0f,
		0.0f,
		1.0f,
		0.0f, // back face
		-1.0f,
		-1.0f,
		-1.0f,
		-1.0f,
		-1.0f,
		1.0f,
		0.0f,
		1.0f,
		0.0f, // left face
		-1.0f,
		-1.0f,
		-1.0f,
		1.0f,
		-1.0f,
		1.0f,
		-1.0f,
		-1.0f,
		1.0f, // base – left front
		1.0f,
		-1.0f,
		1.0f,
		-1.0f,
		-1.0f,
		-1.0f,
		1.0f,
		-1.0f,
		-1.0f // base – right back
	};

	float textureCoods[18]{
		0, 0, 1, 0, .5, 1, 0, 0, 1, 0, .5, 0, 0, 0, 1, 0, 0.5, 1
	};

	TBufferAttribVertexHandle pyramidHandle;
	TDrawVertexHandle textureHandle;
	OTexture Texture;
};

} // namespace Test
