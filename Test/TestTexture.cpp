#include "TestTexture.hpp"

namespace Test
{
OTestTexture::OTestTexture(const OPath& TexturePath, const OPath& ShaderPath, OSharedPtr<RenderAPI::ORenderer> Renderer)
    : Texture(TexturePath), OTest(ShaderPath, Renderer)
{
	pyramidHandle = Renderer->AddAttribBuffer(
	    TVertexContext(new TBuffer(pyramidPositions, sizeof(pyramidPositions)), 0, 3, GL_FLOAT, false, 0, 0, 0));

	textureHandle = CreateVertexElement(
	    TVertexContext(new TBuffer(textureCoods, sizeof(textureCoods)), 1, 2, GL_FLOAT, false, 0, 1, 0),
	    TDrawContext(GL_TRIANGLES, 0, 18, GL_LEQUAL, GL_CCW, GL_DEPTH_TEST));
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
