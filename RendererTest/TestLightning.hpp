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

	OMat4 InvTrTorusMat;
	OMat4 InvTrCubeMat;

	OVec4 GlobalAmbient = { 0.7F, 0.7F, 0.7F, 0.7F };
	SSpotlight SpotLight;

	float SquareVertices;

	uint32 VBO[6];
	uint32 EBO[2];
	uint32 VAO[2];
	OTorus Torus{ 48 };

	OCube Cube;
	SModelContext CubeContext;

	OVec3 RayWorld;
	OMat4 MTorusMatrix = OMat4(1);
	OMat4 CubeMatrix = OMat4(1);
	OMat4 SmallCubeMatrix = OMat4(1);
	OVec4 ShininessContribution = { 1, 1, 1, 1 };
	SModelContext TorusContext;
};

} // namespace RAPI

#endif // RENDERAPI_TESTLIGHTNING_HPP
