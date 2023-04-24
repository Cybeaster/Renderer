#pragma once
#include "Models/Model.hpp"
#include "Test.hpp"
#ifndef RENDERAPI_TESTMODELLING_HPP
#define RENDERAPI_TESTMODELLING_HPP

namespace RAPI
{
class ORenderer;
} // namespace RAPI

namespace RAPI
{

class OTestModelling : public OTest
{
public:
	OTestModelling(const OPath& ShaderPath, ORenderer* Renderer, RAPI::OModel* TestModel, const OPath& ModelTexturePath);
	void OnUpdate(const float& deltaTime, const float& aspect, const OVec3& cameraPos, OMat4& pMat, OMat4& VMat) override;

	RAPI::OModel* Model;
	// temporary without using renderer
	uint32 VAO;
	uint32 NumVBO = 3;
	uint32 VBO[3];
	OTexture ModelTexture;

	SModelContext ModelContext;
	SBufferAttribVertexHandle ModelHandle;
	SDrawVertexHandle TextureHandle;
};

} // namespace RAPI

#endif // RENDERAPI_TESTMODELLING_HPP
