#include "Test.hpp"

#include <Renderer.hpp>
#include <gtc/matrix_transform.hpp>
#include <gtc/type_ptr.hpp>

namespace Test
{
OTest::OTest(TPath shaderPath, TTSharedPtr<RenderAPI::ORenderer> RendererArg)
    : Shader(shaderPath), Renderer(RendererArg)
{
}

void OTest::InitShader(TString shaderPath)
{
	Shader.Init(shaderPath);
}

void OTest::DrawArrays(const TDrawVertexHandle& Handle)
{
	Renderer->DrawArrays(Handle);
}

void OTest::EnableBuffer(const TBufferAttribVertexHandle& Handle)
{
	Renderer->EnableBuffer(Handle);
}

void OTest::EnableBuffer(const TDrawVertexHandle& Handle)
{
	Renderer->EnableBuffer(Handle);
}

TDrawVertexHandle OTest::CreateVertexElement(const TVertexContext& VContext, const TDrawContext& RContext)
{
	return Renderer->CreateVertexElement(VContext, RContext);
}

void OTest::OnUpdate(
    const float deltaTime,
    const float aspect,
    const TVec3& cameraPos,
    TMat4& pMat,
    TMat4& vMat)
{
	Shader.Bind();
	Shader.SetUnformMat4f("proj_matrix", std::move(pMat));
}

} // namespace Test