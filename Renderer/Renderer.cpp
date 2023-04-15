#include "Renderer.hpp"

#include <gtc/matrix_transform.hpp>
#include <gtc/type_ptr.hpp>
#include <gtx/rotate_vector.hpp>
#include <iostream>

#define DEBUG_FPS false

namespace RAPI
{
float ORenderer::Fovy{ 1.0472F };

OMat4 ORenderer::VMat{};
OMat4 ORenderer::PMat{};

OVec2 ORenderer::PressedMousePos{ 0, 0 };
OMat4 ORenderer::MouseCameraRotation{ OMat4(1.F) };

bool ORenderer::RightMousePressed{ false };

/// @brief Mouse Rotation Speed Divide Factor
float ORenderer::MRSDivideFactor{ 100.F };

void ORenderer::Init()
{

	PostInit();
}

void ORenderer::PostInit()
{
	VertexArray.AddVertexArray();
}

void ORenderer::SetInput(OInputHandler* InputHandler)
{
	RenderInputHandler.SetRenderer(this);
	RenderInputHandler.BindKeys(InputHandler);
}

void ORenderer::MoveCamera(const OVec3& Delta)
{
	Camera.Translate(Delta);
}

void ORenderer::RendererStart(float Aspect)
{
	CleanScene();
	CalcPerspective(Aspect);
}

void ORenderer::Tick(const SRenderContext& Context)
{
	RendererStart(Context.DeltaTime);
	for (auto* const test : Tests)
	{
		if (test)
		{
			test->OnUpdate(Context.DeltaTime, Context.AspectRatio, GetCameraPosition(), PMat, VMat);
		}
	}

	RendererEnd();
}

void ORenderer::CalcPerspective(float Aspect) const
{
	PMat = glm::perspective(1.0472f, Aspect, 0.1f, 1000.F); // it needs aspect
	VMat = glm::lookAt(GetCameraPosition() * -1.F, GetCameraPosition(), { 0, 1, 0 });
}

void ORenderer::CleanScene()
{
	GLCall(glClear(GL_COLOR_BUFFER_BIT));
	GLCall(glClear(GL_DEPTH_BUFFER_BIT));
	GLCall(glEnable(GL_CULL_FACE));
}

void ORenderer::AddTest(Test::OTest* testPtr)
{
	if (testPtr)
	{
		testPtr->Init(PMat);
		Tests.push_back(testPtr);
	}
}
SBufferAttribVertexHandle ORenderer::AddAttributeBuffer(const SVertexContext& Context)
{
	return VertexArray.AddAttribBuffer(Context);
}

void ORenderer::EnableBufferAttribArray(const SDrawVertexHandle& Handle)
{
	VertexArray.EnableBufferAttribArray(Handle);
}

void ORenderer::EnableBufferAttribArray(const SBufferAttribVertexHandle& Handle)
{
	VertexArray.EnableBufferAttribArray(Handle);
}

SBufferAttribVertexHandle ORenderer::AddAttribBuffer(const OVertexAttribBuffer& Buffer)
{
	return VertexArray.AddAttribBuffer(Buffer);
}

SBufferAttribVertexHandle ORenderer::AddAttribBuffer(OVertexAttribBuffer&& Buffer)
{
	return VertexArray.AddAttribBuffer(Move(Buffer));
}
SBufferHandle ORenderer::AddBuffer(const void* Data, size_t Size)
{
	return VertexArray.AddBuffer(Data, Size);
}
void ORenderer::BindBuffer(const SBufferHandle& Handle)
{
	VertexArray.BindBuffer(Handle);
}
SBufferHandle ORenderer::AddBuffer(SBufferContext&& Context)
{
	return VertexArray.AddBuffer(Move(Context));
}
SDrawVertexHandle ORenderer::CreateVertexElement(const SVertexContext& VContext, const SDrawContext& RContext)
{
	return VertexArray.CreateVertexElement(VContext, RContext);
}
void ORenderer::Draw(const SDrawVertexHandle& Handle)
{
	VertexArray.Draw(Handle);
}
ORenderer* ORenderer::Get()
{
	if (!Renderer)
	{
		Renderer = new ORenderer();
		return Renderer;
	}

	return Renderer;
}

const OVec3& ORenderer::GetCameraPosition() const
{
	return Camera.GetPosition();
}

void ORenderer::RendererEnd()
{
}

} // namespace RAPI
