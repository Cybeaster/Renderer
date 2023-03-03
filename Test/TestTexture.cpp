#include "TestTexture.hpp"

namespace Test
{
OTestTexture::OTestTexture(const OPath& TexturePath, const OPath& SecondTexturePath, const OPath& ShaderPath, OTSharedPtr<RenderAPI::ORenderer> Renderer)
    : Texture(TexturePath), EarthTexture(SecondTexturePath), OTest(ShaderPath, Renderer)
{
	SModelContext sphereContext;
	Sphere.GetVertexTextureNormalPositions(sphereContext);

	// Pyramid

	pyramidHandle = AddAttribBuffer(
	    RenderAPI::OVertexAttribBuffer(SVertexContext(AddBuffer(pyramidPositions, sizeof(pyramidPositions)), 0, 3, GL_FLOAT, false, 0, 0, nullptr)));

	textureHandle = CreateVertexElement(
	    SVertexContext(AddBuffer(textureCoods, sizeof(textureCoods)), 1, 2, GL_FLOAT, false, 0, 1, nullptr),
	    SDrawContext(GL_TRIANGLES, 0, 18, GL_LEQUAL, GL_CCW, GL_DEPTH_TEST));
	// Pyramid

	// Sphere
	VerticesSphereHandle = AddAttribBuffer(
	    RenderAPI::OVertexAttribBuffer(SVertexContext(AddBuffer(sphereContext.VertexCoords.data(), sphereContext.VertexCoords.size() * 4), 0, 3, GL_FLOAT, false, 0, 0, nullptr)));

	TexturesSphereHandle = CreateVertexElement(
	    SVertexContext(
	        AddBuffer(sphereContext.TextureCoords.data(), sphereContext.TextureCoords.size() * 4), 1, 2, GL_FLOAT, false, 0, 1, nullptr),
	    SDrawContext(
	        GL_TRIANGLES, 0, Sphere.GetNumIndices(), GL_LEQUAL, GL_CCW, GL_DEPTH_TEST));

	NormalsSphereHandle = AddBuffer(sphereContext.NormalsCoords.data(), sphereContext.NormalsCoords.size() * 4);
	// Sphere
	//
	//	// Torus
	//	SModelContext torusContext;
	//	Torus.GetVertexTextureNormalPositions(torusContext);
	//
	//	SBufferContext vertContext(torusContext.VertexCoords.data(),
	//	                           torusContext.VertexCoords.size() * 4,
	//	                           EBufferOptions::StaticDraw,
	//	                           EBufferTypes::ArrayBuffer);
	//
	//	VerticesTorusHandle = AddAttribBuffer(
	//	    RenderAPI::OVertexAttribBuffer(
	//	        SVertexContext(AddBuffer(Move(vertContext)),
	//	                       0,
	//	                       3,
	//	                       GL_FLOAT,
	//	                       false,
	//	                       0,
	//	                       0,
	//	                       nullptr)));
	//
	//	SBufferContext textContext(torusContext.VertexCoords.data(),
	//	                           torusContext.VertexCoords.size() * 4,
	//	                           EBufferOptions::StaticDraw,
	//	                           EBufferTypes::ArrayBuffer);
	//
	//	TexturesTorusHandle = CreateVertexElement(
	//	    SVertexContext(AddBuffer(Move(textContext)),
	//	                   1,
	//	                   2,
	//	                   GL_FLOAT,
	//	                   false,
	//	                   0,
	//	                   1,
	//	                   nullptr),
	//
	//	    SDrawContext(GL_TRIANGLES,
	//	                 0,
	//	                 Sphere.GetNumIndices(),
	//	                 GL_LEQUAL,
	//	                 GL_CCW,
	//	                 GL_DEPTH_TEST));
	//
	//	IndicesTorusHandle = AddBuffer(SBufferContext(Torus.GetIndices().data(),
	//	                         Torus.GetNumIndices() * 4,
	//	                         StaticDraw,
	//	                         ElementArrayBuffer));
	//	// Torus
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

	Texture.Bind(1);
	GetShader().SetUnformMat4f("mv_matrix", glm::translate(VMat, { -2, 0, 0 }));

	EnableAttribArrayBuffer(pyramidHandle);
	EnableAttribArrayBuffer(textureHandle);

	Draw(textureHandle);

	GetShader().SetUnformMat4f("mv_matrix", glm::translate(VMat, { -2, -2, 0 }));
}

} // namespace Test
