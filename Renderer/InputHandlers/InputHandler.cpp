#include "InputHandler.hpp"

#include "Renderer.hpp"

#include <Delegate.hpp>
#include <gtc/matrix_transform.hpp>
#include <gtx/rotate_vector.hpp>
#include <gtx/string_cast.hpp>
#include <iostream>

#define DEBUG_MOUSE_WHEEL false
#define DEBUG_MOUSE_POS true

namespace RenderAPI
{

OTHashMap<EKeys, SKeyState> OInputHandler::KeyMap{};

void OInputHandler::Tick(float DeltaTime)
{
}

void OInputHandler::SetInput(GLFWwindow* Window)
{
	glfwSetWindowSizeCallback(Window, OInputHandler::WindowReshapeCallback);
	glfwSetScrollCallback(Window, OInputHandler::CursorWheelInputCallback);
	glfwSetMouseButtonCallback(Window, OInputHandler::MouseInputCallback);
	glfwSetCursorPosCallback(Window, OInputHandler::MouseCursorMoveCallback);
}

void OInputHandler::WindowReshapeCallback(GLFWwindow* window,
                                          const int& newHeight,
                                          const int& newWidth)
{
	if (!window)
		return;
	glViewport(0, 0, newWidth, newHeight);

	ORenderer::Aspect = static_cast<float>(newWidth / newHeight);
	ORenderer::ScreenWidth = newWidth;
	ORenderer::ScreenHeight = newHeight;
	ORenderer::PMat = glm::perspective(ORenderer::Fovy, ORenderer::Aspect, 0.1F, 1000.F);
}

void OInputHandler::CursorWheelInputCallback(GLFWwindow*  /*window*/, double  /*XOffset*/,
                                             double YOffset)
{
	ORenderer::CameraPos.z -= YOffset;
	if (DEBUG_MOUSE_WHEEL)
	{
		std::cout << glm::to_string(ORenderer::CameraPos) << std::endl;
	}
}

void OInputHandler::MouseInputCallback(GLFWwindow* window, int Button,
                                       int Action, int  /*Mods*/)
{
	if (Button == GLFW_MOUSE_BUTTON_RIGHT)
	{
		if (Action == GLFW_RELEASE)
		{
			ORenderer::RightMousePressed = false;

			double xPos;
			double yPos;
			glfwGetCursorPos(window, &xPos, &yPos);
			ORenderer::PressedMousePos = { xPos, yPos };
		}
		else if (Action == GLFW_PRESS)
		{
			ORenderer::RightMousePressed = true;
		}
	}
}

void OInputHandler::MouseCursorMoveCallback(GLFWwindow*  /*Window*/, double XPos,
                                            double YPos)
{
	if (ORenderer::RightMousePressed)
	{
		auto pos = OVec2(XPos, YPos);
		auto delta = (ORenderer::PressedMousePos - pos);

		ORenderer::CameraPos = glm::rotate(
		    ORenderer::CameraPos, glm::length(delta) / ORenderer::MRSDivideFactor, OVec3(delta.y, delta.x, 0)); // inverted

		ORenderer::PressedMousePos = pos;
		if (DEBUG_MOUSE_POS)
		{
			std::cout << glm::to_string(delta) << std::endl;
		}
	}
}
void OInputHandler::KeyboardInputCallback(GLFWwindow* window, int key,
                                          int scancode, int action, int mods)
{
	if (action == GLFW_PRESS)
	{
		KeyboardInputPressed(window, static_cast<EKeys>(key), scancode, mods);
	}
	else
	{
		KeyboardInputReleased(window, static_cast<EKeys>(key), scancode, mods);
	}
}

void OInputHandler::KeyboardInputPressed(GLFWwindow* /*window*/, EKeys key,
                                         int /*scancode*/, int /*mods*/)
{
	if (KeyMap.contains(key))
	{
		auto state = KeyMap[key];
		state.IsPressed = true;

		state.Callback.Broadcast(state.IsPressed);
	}
}

void OInputHandler::KeyboardInputReleased(GLFWwindow* /*window*/, EKeys key,
                                          int /*scancode*/, int /*mods*/)
{
	if (KeyMap.contains(key))
	{
		auto state = KeyMap[key];
		state.IsPressed = false;

		// state.Callback.Execute(state.IsPressed);
	}
}

} // namespace RenderAPI
