#include "GlfwWindow.hpp"

#include "Application/Application.hpp"
#include "Assert.hpp"

namespace RAPI
{

void OGLFWWindow::InitWindow()
{
	OWindow::InitWindow();

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);

	/* Initialize the library */
	if (!ENSURE(glfwInit()))
	{
		return;
	}
	Height = 720;
	Width = 720;
	/* Create a windowed mode window and its OpenGL context */
	Window = glfwCreateWindow(Width, Height, WindowName.c_str(), nullptr, nullptr);

	if (!ENSURE(Window != nullptr, "Window is not created in glfw!"))
	{
		glfwTerminate();
		return;
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
	int width = 0;
	int height = 0;
	glfwGetFramebufferSize(Window, &width, &height);
	OnReshaped(width, height);
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
	Width = Cast<float>(NewWidth);
	Height = Cast<float>(NewHeight);
	AspectRatio = Width / Height;
	glViewport(0, 0, Width, Height);
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
