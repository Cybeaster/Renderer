
#pragma once
#include "Delegate.hpp"
#include "MulticastDelegate.hpp"
#include "Window.hpp"
#include "glfw3.h"

namespace RAPI
{
class OGLFWWindow : public OWindow
{
public:
	~OGLFWWindow() override;
	GLFWwindow* GetWindow();

	static void WindowReshapeCallback(GLFWwindow* Window, int NewHeight, int NewWidth);

	void OnReshaped(int32 NewWidth, int32 NewHeight);
	void InitWindow() override;

private:
	void CalcDeltaTime() override;
	void CalcAspect() override;
	bool NeedClose() override;
	void DrawStart() override;
	void DrawEnd() override;

private:
	static inline ODelegate<void, int32, int32> OnReshapedDelegate{};
	GLFWwindow* Window;
};

} // namespace RAPI
