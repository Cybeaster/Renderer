#include "Test.hpp"
#include <Renderer.hpp>
#include <gtc/matrix_transform.hpp>
#include <gtc/type_ptr.hpp>
namespace Test
{
    Test::Test(TPath shaderPath, TRenderer *RendererArg) : Shader(shaderPath), Renderer(RendererArg)
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
        pMat = glm::perspective(1.0472f, aspect, 0.01f, 1000.f);
        Shader.SetUnformMat4f("proj_matrix", std::move(pMat));
        vMat = glm::translate(TMat4(1.0f), cameraPos * -1.f);
    }

}