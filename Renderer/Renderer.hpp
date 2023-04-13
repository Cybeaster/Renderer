#pragma once
#include "Camera/Camera.hpp"
#include "Checks/Assert.hpp"
#include "InputHandlers/InputHandler.hpp"
#include "Math.hpp"
#include "SmartPtr.hpp"
#include "Types.hpp"
#include "Utils/Threads/ThreadPool/CustomTestThreadPool/ThreadPool.hpp"
#include "Vector.hpp"
#include "Vertex/VertexArray.hpp"
#include "glfw3.h"

#include <Test.hpp>

class Application;
namespace RAPI
{
/**
 * @brief Singleton class that creates the context, calculates perspective etc.
 *
 */
struct SRenderContext
{
	float DeltaTime{ 0 };
	float AspectRatio{ 0 };
};

class ORenderer
{
public:
	~ORenderer();

	static ORenderer* Get();

	void Tick(const SRenderContext& Context);

	void AddTest(Test::OTest* testPtr);

	SDrawVertexHandle CreateVertexElement(const SVertexContext& VContext, const SDrawContext& RContext);

	void Draw(const SDrawVertexHandle& Handle);

	void EnableBufferAttribArray(const SDrawVertexHandle& Handle);
	void EnableBufferAttribArray(const SBufferAttribVertexHandle& Handle);

	SBufferAttribVertexHandle AddAttribBuffer(const OVertexAttribBuffer& Buffer);
	SBufferAttribVertexHandle AddAttribBuffer(OVertexAttribBuffer&& Buffer);

	SBufferAttribVertexHandle AddAttributeBuffer(const SVertexContext& Context);

	SBufferHandle AddBuffer(const void* Data, size_t Size);
	SBufferHandle AddBuffer(SBufferContext&& Context);

	void BindBuffer(const SBufferHandle& Handle);

	void Init();
	void PostInit();

	void MoveCamera(const OVec3& Delta);

	NODISCARD const OVec3& GetCameraPosition() const;

	static float Fovy;

	static OMat4 VMat;
	static OMat4 PMat;

	static bool RightMousePressed;
	static OVec2 PressedMousePos;

	static OMat4 MouseCameraRotation;
	static float MRSDivideFactor;

	void SetInput(OInputHandler* InputHandler);

private:
	ORenderer() = default;

	void RendererStart(float Aspect);
	void RendererEnd();
	void CleanScene();
	void CalcPerspective(float Aspect) const;

	OCamera Camera;
	ORendererInputHandler RenderInputHandler;
	OVertexArray VertexArray;
	OVector<Test::OTest*> Tests;

	static inline ORenderer* Renderer = nullptr;
};

} // namespace RAPI
