#pragma once
#include "Camera/Camera.hpp"
#include "Checks/Assert.hpp"
#include "InputHandlers/InputHandler.hpp"
#include "InputHandlers/RendererInputHandler.hpp"
#include "Math.hpp"
#include "Models/Model.hpp"
#include "SmartPtr.hpp"
#include "Types.hpp"
#include "Utils/Threads/ThreadPool/CustomTestThreadPool/ThreadPool.hpp"
#include "Vector.hpp"
#include "Vertex/VertexArray.hpp"
#include "glfw3.h"

namespace RAPI
{
class OTest;
/**
 * @brief Singleton class that creates the context, calculates perspective etc.
 *
 */
struct SRenderContext
{
	float DeltaTime{ 0 };
	float AspectRatio{ 0 };
};

class OModel;
class ORenderer
{
public:
	~ORenderer();

	static ORenderer* Get();

	void Tick(const SRenderContext& Context);

	void AddTest(OTest* testPtr);

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

	void AddModel(const SModelContext& Context);

	void Init();
	void PostInit();
	void SetInput(OInputHandler* InputHandler);

	void MoveCamera(ETranslateDirection Dir);
	void SetCameraPosition(OVec3 NewPos);
	void RotateCamera(const OVec2& Delta);

	NODISCARD const OVec3& GetCameraPosition() const;
	NODISCARD const OVec3& GetCameraDirection() const;

	static OMat4 VMat;
	static OMat4 PMat;

private:
	ORenderer() = default;

	void RendererStart(const SRenderContext& Context);
	void RendererEnd();
	void CleanScene();
	void CalcPerspective(float Aspect);

	OCamera Camera;
	ORendererInputHandler RenderInputHandler;
	OVertexArray VertexArray;
	OVector<OTest*> Tests;

	static inline ORenderer* Renderer = nullptr;
};

} // namespace RAPI
