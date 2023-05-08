#include "Test.hpp"

#include <Renderer.hpp>
#include <gtc/matrix_transform.hpp>
#include <gtc/type_ptr.hpp>

namespace RAPI
{
OTest::OTest(const OPath& ShaderPath, RAPI::ORenderer* RendererArg)
    : Shader(ShaderPath), Renderer(RendererArg)
{
}

void OTest::InitShader(const OString& ShaderPath)
{
	Shader.Init(ShaderPath);
}

void OTest::Draw(const SDrawVertexHandle& Handle)
{
	Renderer->Draw(Handle);
}

void OTest::EnableAttribArrayBuffer(const SBufferAttribVertexHandle& Handle)
{
	Renderer->EnableBufferAttribArray(Handle);
}

void OTest::EnableAttribArrayBuffer(const SDrawVertexHandle& Handle)
{
	Renderer->EnableBufferAttribArray(Handle);
}

SDrawVertexHandle OTest::CreateVertexElement(const SVertexContext& VContext, const SDrawContext& RContext)
{
	return Renderer->CreateVertexElement(VContext, RContext);
}

SBufferAttribVertexHandle OTest::AddAttribBuffer(const RAPI::OVertexAttribBuffer& AttribBuffer)
{
	return Renderer->AddAttribBuffer(AttribBuffer);
}

void OTest::OnUpdate(
    const float& /*DeltaTime*/,
    const float& /*Aspect*/,
    const OVec3& /*CameraPos*/,
    OMat4& PMat,
    OMat4& /*VMat*/)
{
	Shader.Bind();
	Shader.SetUniformMat4f("proj_matrix", PMat);
}

SBufferHandle OTest::AddBuffer(const void* Data, size_t Size)
{
	return Renderer->AddBuffer(Data, Size);
}

SBufferHandle OTest::AddBuffer(SBufferContext&& Context)
{
	return Renderer->AddBuffer(Move(Context));
}

void OTest::BindBuffer(const SBufferHandle& Handle)
{
	Renderer->BindBuffer(Handle);
}

} // namespace RAPI