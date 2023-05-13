//
// Created by Cybea on 5/7/2023.
//

#ifndef RENDERAPI_TESTLIGHTNING_HPP
#define RENDERAPI_TESTLIGHTNING_HPP

#include "Models/Torus.hpp"
#include "Test.hpp"
namespace RAPI
{

class OTestLightning : public OTest
{
public:
	OTestLightning(const OPath& TextureFirstPath, const OPath& SecondTexturePath, const OPath& ShaderPath, RAPI::ORenderer* Renderer);
	~OTestLightning() override = default;

	void OnUpdate(
	    const float& DeltaTime,
	    const float& Aspect,
	    const OVec3& CameraPos,
	    OMat4& PMat,
	    OMat4& VMat) override;

private:
	void OnMouseMoved(double NewX, double newY);
	void InstallLights(OMat4 VMatrix);

	OTexture TorusTexture;

	OVec4 CurrentLightPos;
	OVec4 CurrentSecondLightPos;

	OVec3 LightPos;
	OVec3 LightSecPos;

	OMat4 InvTrMat;

	uint32 GlobalAmbLoc, AmbLoc, DiffLoc, SpecLoc, PosLoc, MAmbLoc, MDiffLoc, MSpecLoc, MShiLoc;

	OVec3 InitialLightLoc = { 1, 2, 1 };

	OVec4 GlobalAmbient = { 0.7F, 0.7F, 0.7F, 0.7F };
	OVec4 LightAmbient = { 0, 0, 0, 1.F };
	OVec4 LightDiffuse = { 1.F, 1.F, 1.F, 1.F };
	OVec4 LightSpecular = { 1.F, 1.F, 1.F, 1.F };

	OVec4 LightSecondAmbient = { 0, 0, 0, 1.F };
	OVec4 LightSecondDiffuse = { 0.5F, 0.5F, 0.5F, 0.5F };
	OVec4 LightSecondSpecular = {0.5F, 0.5F, 0.5F, 0.5F };


	uint32 VBO[4];
	uint32 VAO[1];
	OTorus Torus{ 48 };

	OMat4 MTorusMatrix = OMat4(1);
	SModelContext TorusContext;
};

} // namespace RAPI

#endif // RENDERAPI_TESTLIGHTNING_HPP
