#include "TestTexture.hpp"

#include "Assert.hpp"
namespace RAPI
{
OTestTexture::OTestTexture(const OPath& TexturePath, const OPath& SecondTexturePath, const OPath& ShaderPath, RAPI::ORenderer* Renderer, OModel* Model)
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
	    SVertexContext(AddBuffer(sphereContext.Vertices.data(), sphereContext.Vertices.size() * 4), 0, 3, GL_FLOAT, false, 0, 0, nullptr));

	TexturesSphereHandle = CreateVertexElement(
	    SVertexContext(
	        AddBuffer(sphereContext.TexCoords.data(), sphereContext.TexCoords.size() * 4), 1, 2, GL_FLOAT, false, 0, 1, nullptr),
	    SDrawContext(
	        GL_TRIANGLES, 0, Sphere.GetNumIndices(), GL_LEQUAL, GL_CCW, GL_DEPTH_TEST));

	NormalsSphereHandle = AddBuffer(sphereContext.Normals.data(), sphereContext.Normals.size() * 4);
	// Sphere

	//  Torus

	SModelContext torusContext;
	Torus.GetVertexTextureNormalPositions(torusContext);

	SBufferContext vertContext(torusContext.Vertices.data(),
	                           torusContext.Vertices.size() * 4,
	                           EBufferOptions::StaticDraw,
	                           EBufferTypes::ArrayBuffer);

	VerticesTorusHandle = AddAttribBuffer(
	    SVertexContext(AddBuffer(Move(vertContext)), 0, 3, GL_FLOAT, false, 0, 0, nullptr));

	SBufferContext textContext(torusContext.TexCoords.data(),
	                           torusContext.TexCoords.size() * 4,
	                           EBufferOptions::StaticDraw,
	                           EBufferTypes::ArrayBuffer);

	TexturesTorusHandle = AddAttribBuffer(
	    SVertexContext(AddBuffer(Move(textContext)), 1, 2, GL_FLOAT, false, 0, 1, nullptr));

	IndicesTorusHandle = AddBuffer(SBufferContext(Torus.GetIndices().data(),
	                                              Torus.GetIndices().size() * 4,
	                                              StaticDraw,
	                                              ElementArrayBuffer));

	NormalsTorusHandle = AddBuffer(SBufferContext(torusContext.Normals.data(),
	                                              torusContext.Normals.size() * 4,
	                                              StaticDraw,
	                                              ArrayBuffer));
	// Torus

	Model->GetVertexTextureNormalPositions(ImportedModelContext);

	VerticesImportedHandle = CreateVertexElement(
	    SVertexContext(AddBuffer(ImportedModelContext.Vertices.data(), ImportedModelContext.Vertices.size() * 4), 0, 3, GL_FLOAT, false, 0, 0, nullptr),
	    SDrawContext(
	        GL_TRIANGLES, 0, ImportedModelContext.Vertices.size(), GL_LEQUAL, GL_CCW, GL_DEPTH_TEST));

	TexturesImportedHandle = AddAttribBuffer(
	    SVertexContext(
	        AddBuffer(ImportedModelContext.TexCoords.data(), ImportedModelContext.TexCoords.size() * 4), 1, 2, GL_FLOAT, false, 0, 1, nullptr));
}

void OTestTexture::OnUpdate(
    const float& DeltaTime,
    const float& Aspect,
    const OVec3& CameraPos,
    OMat4& PMat,
    OMat4& VMat)
{
	OTest::OnUpdate(DeltaTime, Aspect, CameraPos, PMat, VMat);
	GetShader().SetUniform1i("use_texture", 1);

	EarthTexture.Bind(0);
	GetShader().SetUniformMat4f("mv_matrix", glm::translate(VMat, { 0, 2, -5 }));
	EnableAttribArrayBuffer(VerticesSphereHandle);
	EnableAttribArrayBuffer(TexturesSphereHandle);
	Draw(TexturesSphereHandle);
	EarthTexture.Unbind();

	WallTexture.Bind(0);
	GetShader().SetUniformMat4f("mv_matrix", glm::translate(VMat, { -2, 0, -5 }));
	EnableAttribArrayBuffer(pyramidHandle);
	EnableAttribArrayBuffer(textureHandle);
	Draw(textureHandle);

	GetShader().SetUniformMat4f("mv_matrix", glm::translate(VMat, { -2, -2, -5 }));

	EnableAttribArrayBuffer(VerticesTorusHandle);
	EnableAttribArrayBuffer(TexturesTorusHandle);

	glEnable(GL_CULL_FACE);
	glFrontFace(GL_CCW);

	// TODO make drawing for indices and arrays
	BindBuffer(IndicesTorusHandle);

	GLCall(glDrawElements(GL_TRIANGLES, Torus.GetNumIndices(), GL_UNSIGNED_INT, 0));

	GetShader().SetUniformMat4f("mv_matrix", glm::translate(VMat, { 2, 2, -5 }) * glm::scale(glm::mat4(1.F), { 0.2F, 0.2F, 0.2F }));
	GetShader().SetUniform1i("use_texture", 0);

	EnableAttribArrayBuffer(VerticesImportedHandle);
	EnableAttribArrayBuffer(TexturesImportedHandle);

	GLCall(glDrawArrays(GL_TRIANGLES,
	                    0,
	                    ImportedModelContext.Vertices.size() / 3));

	WallTexture.Unbind();
}

} // namespace RAPI
