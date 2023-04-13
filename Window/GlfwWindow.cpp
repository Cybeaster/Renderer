#include "GlfwWindow.hpp"

#include "Assert.hpp"

namespace RAPI
{

void OGLFWWindow::InitWindow()
{
	/* Initialize the library */
	ASSERT(glfwInit());

	/* Create a windowed mode window and its OpenGL context */
	Window = glfwCreateWindow(Width, Height, WindowName.c_str(), nullptr, nullptr);

	if (ENSURE(!Window == false, "Window is not created in glfw!"))
	{
		glfwTerminate();
	}

	/* Make the window's context current */
	glfwMakeContextCurrent(Window);
	glfwSwapInterval(1);

	ASSERT(glewInit() == GLEW_OK)

	GLCall(glEnable(GL_BLEND));
	GLCall(glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA));

	OnReshapedDelegate.BindRaw(this, &OGLFWWindow::OnReshaped);
	glfwSetWindowSizeCallback(Window, OGLFWWindow::WindowReshapeCallback);
}

bool OGLFWWindow::NeedClose()
{
	return !!glfwWindowShouldClose(Window);
}

void OGLFWWindow::CalcDeltaTime()
{
	auto currentTime = glfwGetTime();
	DeltaTime = currentTime - PreviousTime;
	PreviousTime = currentTime;
}

void OGLFWWindow::DrawStart()
{
	CalcAspect();
	CalcDeltaTime();
}

void OGLFWWindow::DrawEnd()
{
	/* Swap front and back buffers */
	glfwSwapBuffers(Window);
	glfwPollEvents();
}

void OGLFWWindow::CalcAspect()
{
	glfwGetFramebufferSize(Window, &Width, &Height);
	AspectRatio = static_cast<float>(static_cast<float>(Width) / static_cast<float>(Height));
}

OGLFWWindow::~OGLFWWindow()
{
	glfwDestroyWindow(Window);
	glfwTerminate();
}

GLFWwindow* OGLFWWindow::GetWindow()
{
	return Window;
}

void OGLFWWindow::OnReshaped(int32 NewWidth, int32 NewHeight)
{
	Width = NewWidth;
	Height = NewHeight;
	AspectRatio = static_cast<float>(static_cast<float>(Width) / static_cast<float>(Height));
}

void OGLFWWindow::WindowReshapeCallback(GLFWwindow* Window, int NewHeight, int NewWidth)
{
	if (!Window)
	{
		return;
	}

	OGLFWWindow::OnReshapedDelegate.Execute(NewWidth, NewHeight);
}

} // namespace RAPI
