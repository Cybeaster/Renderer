#include "InputHandler.hpp"

#include "Externals/Include/glm/gtc/matrix_transform.hpp"
#include "Externals/Include/glm/gtx/rotate_vector.hpp"
#include "Externals/Include/glm/gtx/string_cast.hpp"
#include "InputHandlers/InputHandler.hpp"
#include "Renderer.hpp"
#include "Utils/Delegate/Delegate.hpp"
#include "glfw3.h"

#include <iostream>

#define DEBUG_MOUSE_WHEEL false
#define DEBUG_MOUSE_POS true

namespace RAPI
{

OHashMap<EKeys, SKeyState> OInputHandler::KeyMap{};

OInputHandler::OInputHandler(GLFWwindow* Window, bool Enable)
{
	if (Enable)
	{
		InitHandlerWith(Window);
	}
}

void OInputHandler::InitHandlerWith(GLFWwindow* Window)
{
	SetInput(Window);
}

void OInputHandler::Tick(float DeltaTime)
{
}

void OInputHandler::SetInput(GLFWwindow* Window)
{
	glfwSetScrollCallback(Window, OInputHandler::CursorWheelInputCallback);
	glfwSetMouseButtonCallback(Window, OInputHandler::MouseInputCallback);
	glfwSetCursorPosCallback(Window, OInputHandler::MouseCursorMoveCallback);
	glfwSetKeyCallback(Window, OInputHandler::KeyboardInputCallback);
}

void OInputHandler::CursorWheelInputCallback(GLFWwindow* /*window*/, double /*XOffset*/,
                                             double YOffset)
{
	ORenderer::Get()->MoveCamera({ 0, 0, YOffset });
	if (DEBUG_MOUSE_WHEEL)
	{
		std::cout << glm::to_string(ORenderer::Get()->GetCameraPosition()) << std::endl;
	}
}

void OInputHandler::MouseInputCallback(GLFWwindow* window, int Button,
                                       int Action, int /*Mods*/)
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

void OInputHandler::MouseCursorMoveCallback(GLFWwindow* /*Window*/, double XPos,
                                            double YPos)
{
	if (ORenderer::RightMousePressed)
	{
		auto pos = OVec2(XPos, YPos);
		auto delta = (ORenderer::PressedMousePos - pos);

		// TODO Rot camera appropriately
		// ORenderer::CameraPos = glm::rotate(
		//    ORenderer::CameraPos, glm::length(delta) / ORenderer::MRSDivideFactor, OVec3(delta.y, delta.x, 0)); // inverted
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
	if (action == GLFW_PRESS || action == GLFW_REPEAT)
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

		state.Callback.Broadcast(state.IsPressed);
	}
}
} // namespace RAPI
