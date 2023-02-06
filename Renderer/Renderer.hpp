#pragma once
#include "Checks/Assert.hpp"
#include "InputHandlers/InputHandler.hpp"
#include "InputHandlers/RendererInputHandler.hpp"
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

class TRenderer
{
public:
	static auto GetRenderer()
	{
		if (!SingletonRenderer)
		{
			SingletonRenderer = TTSharedPtr<TRenderer>(new TRenderer());
			return SingletonRenderer;
		}
		else
		{
			return SingletonRenderer;
		}
	}

	/**
	 * @brief Initalizes glfw Opengl context and creates a window.
	 *
	 * @return GLFWwindow*
	 */
	GLFWwindow* GLFWInit();
	void GLFWRenderTickStart();

	void AddTest(Test::OTest* testPtr);

	inline TVector<Test::Test*>& getTests()
	{
		return Tests;
	}

	TDrawVertexHandle CreateVertexElement(const TVertexContext& VContext, const TDrawContext& RContext)
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

	void EnableBuffer(const TBufferAttribVertexHandle& Handle)
	{
		VertexArray.EnableBuffer(Handle);
	}

	TBufferAttribVertexHandle AddAttribBuffer(const TVertexAttribBuffer& Buffer)
	{
		return VertexArray.AddAttribBuffer(Buffer);
	}

	TBufferAttribVertexHandle AddAttributeBuffer(const TVertexContext& Context)
	{
		return VertexArray.AddAttribBuffer(Context);
	}
	void TranslateCameraLocation(const glm::mat4& Transform);
	void LookAtCamera(const TVec3& Position);

	void Init();
	void PostInit();

	static int ScreenWidth;
	static int ScreenHeight;

	static float Aspect;
	static float DeltaTime;
	static float LastFrame;
	static float CurrentFrame;
	static float Fovy;
	static TVec3 CameraPos;

	static TMat4 VMat;
	static TMat4 PMat;

	static bool RightMousePressed;
	static TVec2 PressedMousePos;

	static TMat4 MouseCameraRotation;
	static float MRSDivideFactor;

	~TRenderer();

	void MoveCamera(const TVec3& Delta);

private:
	TRenderer() = default;

	void GLFWRendererStart(float currentTime);
	void GLFWRendererEnd();
	void CalcDeltaTime(float currentTime);
	void CleanScene();
	void GLFWCalcPerspective(GLFWwindow* window);
	void PrintDebugInfo();
	void CalcScene();
	void SetInput();

	Thread::TThreadPool RendererThreadPool;
	TVertexArray VertexArray;

	TInputHandler InputHandler;

	GLFWwindow* Window;
	TVector<Test::Test*> Tests;
	static inline TTSharedPtr<TRenderer> SingletonRenderer = nullptr;
};

} // namespace RenderAPI
