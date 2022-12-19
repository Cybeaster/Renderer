#include "Test.hpp"
#include <Renderer.hpp>
#include <gtc/matrix_transform.hpp>
#include <gtc/type_ptr.hpp>
namespace Test
{
    Test::Test(TString shaderPath, TRenderer *RendererArg) : shader(shaderPath), Renderer(RendererArg)
    {
    }

    Test::~Test()
    {
    }

    void Test::AddBuffer(void *buffer, int32_t size)
    {
        buffers.push_back(std::make_shared<TBuffer>(buffer, size));
    }

    void Test::InitShader(TString shaderPath)
    {
        shader.Init(shaderPath);
    }

    void Test::EnableVertexArray(GLuint bufferID)
    {
        buffers[bufferID]->Bind();
        GLCall(glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0));
        GLCall(glEnableVertexAttribArray(0));
    }

    void Test::DrawBuffer(const TVertexArrayHandle &Handle)
    {
        Renderer->DrawBuffer(Handle);
    }

    TVertexArrayHandle Test::CreateVertexElement(const TVertexContext &VContext, const TDrawContext &RContext)
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
        shader.Bind();
        pMat = glm::perspective(1.0472f, aspect, 0.01f, 1000.f);
        shader.SetUnformMat4f("proj_matrix", std::move(pMat));
        vMat = glm::translate(TMat4(1.0f), cameraPos * -1.f);
    }

}