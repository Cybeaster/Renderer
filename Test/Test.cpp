#include "Test.hpp"

#include <Renderer.hpp>
#include <gtc/matrix_transform.hpp>
#include <gtc/type_ptr.hpp>

namespace Test
{
OTest::OTest(const OPath& ShaderPath, const OSharedPtr<RenderAPI::ORenderer>& RendererArg)
    : Shader(ShaderPath), Renderer(RendererArg)
{
}

void OTest::InitShader(const OString& ShaderPath)
{
	Shader.Init(ShaderPath);
}

void OTest::DrawArrays(const TDrawVertexHandle& Handle)
{
	Renderer->DrawArrays(Handle);
}

void OTest::EnableBuffer(const OBufferAttribVertexHandle& Handle)
{
	Renderer->EnableBuffer(Handle);
}

void OTest::EnableBuffer(const TDrawVertexHandle& Handle)
{
	Renderer->EnableBuffer(Handle);
}

TDrawVertexHandle OTest::CreateVertexElement(const SVertexContext& VContext, const SDrawContext& RContext)
{
	return Renderer->CreateVertexElement(VContext, RContext);
}

void OTest::OnUpdate(
    const float& /*DeltaTime*/,
    const float& /*Aspect*/,
    const OVec3& /*CameraPos*/,
    OMat4& PMat,
    OMat4& /*VMat*/)
{
	Shader.Bind();
	Shader.SetUnformMat4f("proj_matrix", Move(PMat));
}

} // namespace Test