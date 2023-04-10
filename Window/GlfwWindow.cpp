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

double OGLFWWindow::GetDeltaTime() const
{
	return DeltaTime;
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

} // namespace RAPI
