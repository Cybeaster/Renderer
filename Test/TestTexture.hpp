#pragma once
#include "../Renderer/Vertex/SimpleVertexHandle.hpp"

#include <Test.hpp>
#include <Texture.hpp>

class GLFWwindow;
namespace Test
{
class OTestTexture : public OTest
{
public:
	OTestTexture(const TPath& filePath, const TPath& ShaderPath, OSharedPtr<RenderAPI::ORenderer> Renderer);
	~OTestTexture();

	virtual void OnUpdate(
	    const float deltaTime,
	    const float aspect,
	    const TVec3& cameraPos,
	    TMat4& pMat,
	    TMat4& vMat) override;

private:
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

	float textureCoods[18]{
		0, 0, 1, 0, .5, 1, 0, 0, 1, 0, .5, 0, 0, 0, 1, 0, 0.5, 1
	};

	TBufferAttribVertexHandle pyramidHandle;
	TDrawVertexHandle textureHandle;
	OTexture Texture;
};

} // namespace Test
