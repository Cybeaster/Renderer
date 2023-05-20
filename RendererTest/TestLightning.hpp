//
// Created by Cybea on 5/7/2023.
//

#ifndef RENDERAPI_TESTLIGHTNING_HPP
#define RENDERAPI_TESTLIGHTNING_HPP

#include "Light/LightTypes.hpp"
#include "Models/Cube.hpp"
#include "Models/Plane.hpp"
#include "Models/Torus.hpp"
#include "Test.hpp"
namespace RAPI
{

class OTestLightning : public OTest
{
public:
	OTestLightning(const OPath& TextureFirstPath, const OPath& SecondTexturePath, const OPath& ShaderPath, RAPI::ORenderer* Renderer, const OPath& ShadowShaderPath);
	~OTestLightning() override = default;

	void OnUpdate(
	    const float& DeltaTime,
	    const float& Aspect,
	    const OVec3& CameraPos,
	    OMat4& PMat,
	    OMat4& VMat) override;

private:
	void ComputeShadows(float Aspect, OVec3& LightDir, OVec3& LightPos);

	OVec3 ComputeRayView(const OMat4& PMat, const OMat4& VMat);
	void ComputeLight(const OMat4& VMat, const OVec3& CameraPos, const OMat4& PMat);
	void CalcModelMatrices();
	void SetupNormalAndMVMatrices(const OMat4& Normal, const OMat4& MV);
	void OnMouseMoved(double NewX, double newY);
	void InstallLights(OMat4 VMatrix);
	void SetupShadowBuffers();

	void DrawModelVertices(uint32 VAOIdx, const OMat4& ModelMatrix, const OMat4& VMat, const SModelContext& Context);
	void DrawModelIndices(uint32 VAOIdx, const OMat4& ModelMatrix, const OMat4& VMat, uint32 NumIndices);

	OTexture TorusTexture;

	OMat4 InvTrTorusMat;
	OMat4 InvTrCubeMat;

	OVec4 GlobalAmbient;
	SSpotlight SpotLight;

	OMat4 BiasesMat = OMat4(0.5, 0, 0, 0.5,
	                        0, 0.5, 0, 0.5,
	                        0, 0, 0.5, 0.5,
	                        0, 0, 0, 1);

	uint32 VBO[6];
	uint32 EBO[2];
	uint32 VAO[2];
	OTorus Torus{ 48 };

	OCube Cube;

	OMat4 DownCubeMat = OMat4(1);

	SModelContext CubeContext;

	OVec3 RayWorld;
	OMat4 MTorusMatrix = OMat4(1);
	OMat4 CubeMatrix = OMat4(1);
	OMat4 SmallCubeMatrix = OMat4(1);
	OVec4 ShininessContribution = { 1, 1, 1, 1 };
	SModelContext TorusContext;

	OShader ShadowShader;

	uint32 ShadowBuffer;
	uint32 ShadowTexture;

	OMat4 SpotLightPVMat;
};

} // namespace RAPI

#endif // RENDERAPI_TESTLIGHTNING_HPP
