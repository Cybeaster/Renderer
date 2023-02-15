#include "TestTexture.hpp"

namespace Test
{
OTestTexture::OTestTexture(const OPath& TexturePath, const OPath& ShaderPath, OTSharedPtr<RenderAPI::ORenderer> Renderer)
    : Texture(TexturePath), OTest(ShaderPath, Renderer)
{
	pyramidHandle = Renderer->AddAttribBuffer(
	    SVertexContext(new OBuffer(pyramidPositions, sizeof(pyramidPositions)), 0, 3, GL_FLOAT, false, 0, 0, 0));

	textureHandle = CreateVertexElement(
	    SVertexContext(new OBuffer(textureCoods, sizeof(textureCoods)), 1, 2, GL_FLOAT, false, 0, 1, 0),
	    SDrawContext(GL_TRIANGLES, 0, 18, GL_LEQUAL, GL_CCW, GL_DEPTH_TEST));
}

void OTestTexture::OnUpdate(
    const float& DeltaTime,
    const float& Aspect,
    const OVec3& CameraPos,
    OMat4& PMat,
    OMat4& VMat)
{
	OTest::OnUpdate(DeltaTime, Aspect, CameraPos, PMat, VMat);

	GetShader().SetUnformMat4f(
	    "mv_matrix",
	    VMat);

	EnableBuffer(pyramidHandle);
	EnableBuffer(textureHandle);

	Texture.Bind(0);

	DrawArrays(textureHandle);
}
} // namespace Test
