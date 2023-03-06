#pragma once
#include "Camera/Camera.hpp"
#include "Checks/Assert.hpp"
#include "InputHandlers/InputHandler.hpp"
#include "Math.hpp"
#include "SmartPtr.hpp"
#include "ThreadPool.hpp"
#include "Types.hpp"
#include "Vector.hpp"
#include "Vertex/VertexArray.hpp"
#include "glfw3.h"

#include <Test.hpp>

struct GLFWwindow;
class Application;
namespace RenderAPI
{
/**
 * @brief Singleton class that creates the context, calculates perspective, frames etc.
 *
 */

class ORenderer
{
public:
	~ORenderer();

	static auto Get()
	{
		if (!SingletonRenderer)
		{
			SingletonRenderer = OTSharedPtr<ORenderer>(new ORenderer());
			return SingletonRenderer;
		}

		return SingletonRenderer;
	}

	/**
	 * @brief Initalizes glfw Opengl context and creates a window.
	 *
	 * @return GLFWwindow*
	 */
	GLFWwindow* GLFWInit();
	void GLFWRenderTickStart();

	void AddTest(Test::OTest* testPtr);

	inline OTVector<Test::OTest*>& GetTests()
	{
		return Tests;
	}

	SDrawVertexHandle CreateVertexElement(const SVertexContext& VContext, const SDrawContext& RContext)
	{
		return VertexArray.CreateVertexElement(VContext, RContext);
	}

	void Draw(const SDrawVertexHandle& Handle)
	{
		VertexArray.Draw(Handle);
	}

	void EnableBufferAttribArray(const SDrawVertexHandle& Handle);

	void EnableBufferAttribArray(const SBufferAttribVertexHandle& Handle);

	SBufferAttribVertexHandle AddAttribBuffer(const OVertexAttribBuffer& Buffer);
	SBufferAttribVertexHandle AddAttribBuffer(OVertexAttribBuffer&& Buffer);
	SBufferAttribVertexHandle AddAttributeBuffer(const SVertexContext& Context);

	SBufferHandle AddBuffer(const void* Data, size_t Size);
	SBufferHandle AddBuffer(SBufferContext&& Context);

	void BindBuffer(const SBufferHandle& Handle);

	void TranslateCameraLocation(const glm::mat4& Transform);
	void LookAtCamera(const OVec3& Position);

	void Init();
	void PostInit();

	void MoveCamera(const OVec3& Delta);

	const OVec3& GetCameraPosition() const
	{
		return Camera.GetPosition();
	}

	FORCEINLINE GLFWwindow* GetWindowContext() const
	{
		return Window;
	}

	static int ScreenWidth;
	static int ScreenHeight;

	static float Aspect;
	static float DeltaTime;
	static float LastFrame;
	static float CurrentFrame;
	static float Fovy;

	static OMat4 VMat;
	static OMat4 PMat;

	static bool RightMousePressed;
	static OVec2 PressedMousePos;

	static OMat4 MouseCameraRotation;
	static float MRSDivideFactor;

private:
	ORenderer() = default;

	void GLFWRendererStart(float currentTime);
	void GLFWRendererEnd();
	void CalcDeltaTime(float currentTime);
	void CleanScene();
	void GLFWCalcPerspective(GLFWwindow* window);
	void PrintDebugInfo();
	void CalcScene();
	void SetInput();

	float InputStepOffset = 0.1F;

	OCamera Camera;

	Thread::OThreadPool RendererThreadPool;

	OVertexArray VertexArray;
	OInputHandler InputHandler{ this };
	GLFWwindow* Window;
	OTVector<Test::OTest*> Tests;

	static inline OTSharedPtr<ORenderer> SingletonRenderer = nullptr;
};

} // namespace RenderAPI
