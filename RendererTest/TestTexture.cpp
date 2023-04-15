#include "TestTexture.hpp"

namespace Test
{
OTestTexture::OTestTexture(const OPath& TexturePath, const OPath& SecondTexturePath, const OPath& ShaderPath, RAPI::ORenderer* Renderer)
    : WallTexture(TexturePath), EarthTexture(SecondTexturePath), OTest(ShaderPath, Renderer)
{
	SModelContext sphereContext;
	Sphere.GetVertexTextureNormalPositions(sphereContext);

	// Pyramid

	pyramidHandle = AddAttribBuffer(
	    SVertexContext(AddBuffer(pyramidPositions, sizeof(pyramidPositions)), 0, 3, GL_FLOAT, false, 0, 0, nullptr));

	textureHandle = CreateVertexElement(
	    SVertexContext(AddBuffer(textureCoods, sizeof(textureCoods)), 1, 2, GL_FLOAT, false, 0, 1, nullptr),
	    SDrawContext(GL_TRIANGLES, 0, 18, GL_LEQUAL, GL_CCW, GL_DEPTH_TEST));
	// Pyramid

	// Sphere
	VerticesSphereHandle = AddAttribBuffer(
	    SVertexContext(AddBuffer(sphereContext.VertexCoords.data(), sphereContext.VertexCoords.size() * 4), 0, 3, GL_FLOAT, false, 0, 0, nullptr));

	TexturesSphereHandle = CreateVertexElement(
	    SVertexContext(
	        AddBuffer(sphereContext.TextureCoords.data(), sphereContext.TextureCoords.size() * 4), 1, 2, GL_FLOAT, false, 0, 1, nullptr),
	    SDrawContext(
	        GL_TRIANGLES, 0, Sphere.GetNumIndices(), GL_LEQUAL, GL_CCW, GL_DEPTH_TEST));

	NormalsSphereHandle = AddBuffer(sphereContext.NormalsCoords.data(), sphereContext.NormalsCoords.size() * 4);
	// Sphere

	//  Torus
	SModelContext torusContext;
	Torus.GetVertexTextureNormalPositions(torusContext);

	SBufferContext vertContext(torusContext.VertexCoords.data(),
	                           torusContext.VertexCoords.size() * 4,
	                           EBufferOptions::StaticDraw,
	                           EBufferTypes::ArrayBuffer);

	VerticesTorusHandle = AddAttribBuffer(
	    SVertexContext(AddBuffer(Move(vertContext)), 0, 3, GL_FLOAT, false, 0, 0, nullptr));

	SBufferContext textContext(torusContext.TextureCoords.data(),
	                           torusContext.TextureCoords.size() * 4,
	                           EBufferOptions::StaticDraw,
	                           EBufferTypes::ArrayBuffer);

	TexturesTorusHandle = AddAttribBuffer(
	    SVertexContext(AddBuffer(Move(textContext)), 1, 2, GL_FLOAT, false, 0, 1, nullptr));

	IndicesTorusHandle = AddBuffer(SBufferContext(Torus.GetIndices().data(),
	                                              Torus.GetIndices().size() * 4,
	                                              StaticDraw,
	                                              ElementArrayBuffer));

	NormalsTorusHandle = AddBuffer(SBufferContext(torusContext.NormalsCoords.data(),
	                                              torusContext.NormalsCoords.size() * 4,
	                                              StaticDraw,
	                                              ArrayBuffer));
	// Torus
}

void OTestTexture::OnUpdate(
    const float& DeltaTime,
    const float& Aspect,
    const OVec3& CameraPos,
    OMat4& PMat,
    OMat4& VMat)
{
	OTest::OnUpdate(DeltaTime, Aspect, CameraPos, PMat, VMat);

	EarthTexture.Bind(0);

	GetShader().SetUnformMat4f("mv_matrix", glm::translate(VMat, { 0, 2, 0 }));

	EnableAttribArrayBuffer(VerticesSphereHandle);
	EnableAttribArrayBuffer(TexturesSphereHandle);

	Draw(TexturesSphereHandle);
	EarthTexture.Unbind();

	WallTexture.Bind(0);
	GetShader().SetUnformMat4f("mv_matrix", glm::translate(VMat, { -2, 0, 0 }));

	EnableAttribArrayBuffer(pyramidHandle);
	EnableAttribArrayBuffer(textureHandle);

	Draw(textureHandle);

	GetShader().SetUnformMat4f("mv_matrix", glm::translate(VMat, { -2, -2, 0 }));

	EnableAttribArrayBuffer(VerticesTorusHandle);
	EnableAttribArrayBuffer(TexturesTorusHandle);

	glEnable(GL_CULL_FACE);
	glFrontFace(GL_CCW);

	// TODO make drawing for indices and arrays
	BindBuffer(IndicesTorusHandle);

	glDrawElements(GL_TRIANGLES, Torus.GetNumIndices(), GL_UNSIGNED_INT, 0);

	WallTexture.Unbind();
}

} // namespace Test
