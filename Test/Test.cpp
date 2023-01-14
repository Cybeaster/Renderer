#include "Test.hpp"
#include <Renderer.hpp>
#include <gtc/matrix_transform.hpp>
#include <gtc/type_ptr.hpp>
namespace Test
{
    Test::Test(TPath shaderPath, TTSharedPtr<RenderAPI::TRenderer> RendererArg) : Shader(shaderPath), Renderer(RendererArg)
    {
    }

    Test::~Test()
    {
    }

    void Test::InitShader(TString shaderPath)
    {
        Shader.Init(shaderPath);
    }

    void Test::DrawArrays(const TDrawVertexHandle &Handle)
    {
        Renderer->DrawArrays(Handle);
    }

    void Test::EnableBuffer(const TBufferAttribVertexHandle &Handle)
    {
        Renderer->EnableBuffer(Handle);
    }

    void Test::EnableBuffer(const TDrawVertexHandle &Handle)
    {
        Renderer->EnableBuffer(Handle);
    }

    TDrawVertexHandle Test::CreateVertexElement(const TVertexContext &VContext, const TDrawContext &RContext)
    {
        return Renderer->CreateVertexElement(VContext, RContext);
    }

    void Test::OnUpdate(
        const float deltaTime,
        const float aspect,
        const TVec3 &cameraPos,
        TMat4 &pMat,
        TMat4 &vMat)
    {
        Shader.Bind();
        Shader.SetUnformMat4f("proj_matrix", std::move(pMat));
    }

}