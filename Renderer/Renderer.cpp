#include "Renderer.hpp"

#include "Test.hpp"

#include <gtc/matrix_transform.hpp>
#include <gtc/type_ptr.hpp>
#include <gtx/rotate_vector.hpp>
#include <iostream>

#define DEBUG_FPS false

namespace RAPI
{

OMat4 ORenderer::VMat{};
OMat4 ORenderer::PMat{};

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

void ORenderer::MoveCamera(ETranslateDirection Dir)
{
	Camera.Translate(Dir);
}

void ORenderer::RendererStart(const SRenderContext& Context)
{
	CleanScene();
	CalcPerspective(Context.AspectRatio);
}

void ORenderer::Tick(const SRenderContext& Context)
{
	RendererStart(Context);
	for (auto* test : Tests)
	{
		if (test)
		{
			test->OnUpdate(Context.DeltaTime, Context.AspectRatio, GetCameraPosition(), PMat, VMat);
		}
	}

	RendererEnd();
}

void ORenderer::CalcPerspective(float Aspect)
{
	PMat = glm::perspective(1.0472f, Aspect, 0.1f, 1000.F);
	VMat = Camera.GetCameraView();
}

void ORenderer::CleanScene()
{
	GLCall(glClear(GL_COLOR_BUFFER_BIT));
	GLCall(glClear(GL_DEPTH_BUFFER_BIT));
	GLCall(glEnable(GL_CULL_FACE));
}

void ORenderer::AddTest(OTest* testPtr)
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
void ORenderer::RotateCamera(const OVec2& Delta)
{
	Camera.Rotate(Delta.x, Delta.y);
}

} // namespace RAPI
