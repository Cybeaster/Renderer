#include "TestTexture.hpp"

namespace Test
{
    TestTexture::TestTexture(const TPath &TexturePath, const TPath &ShaderPath, TTSharedPtr<RenderAPI::TRenderer> Renderer)
        : Texture(TexturePath), Test(ShaderPath, Renderer)
    {

        pyramidHandle = Renderer->AddAttribBuffer(
            TVertexContext(new TBuffer(pyramidPositions, sizeof(pyramidPositions)), 0, 3, GL_FLOAT, false, 0, 0, 0));

        textureHandle = CreateVertexElement(
            TVertexContext(new TBuffer(textureCoods, sizeof(textureCoods)), 1, 2, GL_FLOAT, false, 0, 1, 0),
            TDrawContext(GL_TRIANGLES, 0, 18, GL_LEQUAL, GL_CCW, GL_DEPTH_TEST));
    }

    void TestTexture::OnUpdate(
        const float deltaTime,
        const float aspect,
        const TVec3 &cameraPos,
        TMat4 &pMat,
        TMat4 &vMat)
    {
        Test::OnUpdate(deltaTime, aspect, cameraPos, pMat, vMat);

        GetShader().SetUnformMat4f(
            "mv_matrix",
            vMat);

        EnableBuffer(pyramidHandle);
        EnableBuffer(textureHandle);
        
        Texture.Bind(0);

        DrawArrays(textureHandle);
    }

    TestTexture::~TestTexture()
    {
    }

} // namespace test
