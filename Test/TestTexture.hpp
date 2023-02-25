#pragma once
#include "../Renderer/Vertex/SimpleVertexHandle.hpp"
#include "Models/Sphere.hpp"

#include <Test.hpp>
#include <Texture.hpp>

class GLFWwindow;
namespace Test
{
class OTestTexture : public OTest
{
public:
	OTestTexture(const OPath& filePath, const OPath& ShaderPath, OTSharedPtr<RenderAPI::ORenderer> Renderer);
	~OTestTexture() override = default;

	void OnUpdate(
	    const float& DeltaTime,
	    const float& Aspect,
	    const OVec3& CameraPos,
	    OMat4& PMat,
	    OMat4& VMat) override;

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

	RenderAPI::OSphere Sphere{ 48 };

	OBufferAttribVertexHandle VerticesSphereHandle;
	OBufferAttribVertexHandle NormalsSphereHandle;
	OBufferAttribVertexHandle TexturesSphereHandle;

	OBufferAttribVertexHandle pyramidHandle;
	TDrawVertexHandle textureHandle;
	OTexture Texture;
};

} // namespace Test
