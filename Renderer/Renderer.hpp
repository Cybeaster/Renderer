#pragma once
#include "Checks/Assert.hpp"
#include "InputHandlers/InputHandler.hpp"
#include "Math.hpp"
#include "SmartPtr.hpp"
#include "ThreadPool.hpp"
#include "Types.hpp"
#include "Vector.hpp"
#include "Vertex/VertexArray.hpp"

#include <Test.hpp>

#define GLCall(x)   \
	GLClearError(); \
	x;              \
	ASSERT(GLLogCall(#x, __FILE__, __LINE__))

void GLClearError();
bool GLLogCall(const char* func, const char* file, int line);

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
	static auto GetRenderer()
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

	TDrawVertexHandle CreateVertexElement(const SVertexContext& VContext, const SDrawContext& RContext)
	{
		return VertexArray.CreateVertexElement(VContext, RContext);
	}

	void DrawArrays(const TDrawVertexHandle& Handle)
	{
		VertexArray.DrawArrays(Handle);
	}

	void EnableBuffer(const TDrawVertexHandle& Handle)
	{
		VertexArray.EnableBuffer(Handle);
	}

	void EnableBuffer(const OBufferAttribVertexHandle& Handle)
	{
		VertexArray.EnableBuffer(Handle);
	}

	OBufferAttribVertexHandle AddAttribBuffer(const TVertexAttribBuffer& Buffer)
	{
		return VertexArray.AddAttribBuffer(Buffer);
	}

	OBufferAttribVertexHandle AddAttributeBuffer(const SVertexContext& Context)
	{
		return VertexArray.AddAttribBuffer(Context);
	}
	void TranslateCameraLocation(const glm::mat4& Transform);
	void LookAtCamera(const OVec3& Position);

	void Init();
	void PostInit();

	static int ScreenWidth;
	static int ScreenHeight;

	static float Aspect;
	static float DeltaTime;
	static float LastFrame;
	static float CurrentFrame;
	static float Fovy;
	static OVec3 CameraPos;

	static OMat4 VMat;
	static OMat4 PMat;

	static bool RightMousePressed;
	static OVec2 PressedMousePos;

	static OMat4 MouseCameraRotation;
	static float MRSDivideFactor;

	~ORenderer();

	void MoveCamera(const OVec3& Delta);

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

	Thread::OThreadPool RendererThreadPool;

	OVertexArray VertexArray;
	OInputHandler InputHandler;

	GLFWwindow* Window;
	OTVector<Test::OTest*> Tests;
	static inline OTSharedPtr<ORenderer> SingletonRenderer = nullptr;
};

} // namespace RenderAPI
