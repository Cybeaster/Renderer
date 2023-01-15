#include "InputHandler.hpp"
#include "Renderer.hpp"
#include <gtx/string_cast.hpp>
#include <gtc/matrix_transform.hpp>
#include <gtx/rotate_vector.hpp>
#include <Delegate.hpp>

#define DEBUG_MOUSE_WHEEL false
#define DEBUG_MOUSE_POS true

namespace RenderAPI
{

    TInputHandler::~TInputHandler()
    {
    }

    void TInputHandler::Tick(float DeltaTime)
    {
    }

    void TInputHandler::SetInput(GLFWwindow *Window)
    {
        glfwSetWindowSizeCallback(Window, TInputHandler::WindowReshapeCallback);
        glfwSetScrollCallback(Window, TInputHandler::CursorWheelInputCallback);
        glfwSetMouseButtonCallback(Window, TInputHandler::MouseInputCallback);
        glfwSetCursorPosCallback(Window, TInputHandler::MouseCursorMoveCallback);

        
    }

    void TInputHandler::WindowReshapeCallback(GLFWwindow *window, const int newHeight, const int newWidth)
    {
        if (!window)
            return;
        glViewport(0, 0, newWidth, newHeight);

        TRenderer::Aspect = static_cast<float>(newWidth / newHeight);
        TRenderer::ScreenWidth = newWidth;
        TRenderer::ScreenHeight = newHeight;
        TRenderer::PMat = glm::perspective(TRenderer::Fovy, TRenderer::Aspect, 0.1f, 1000.f);
    }

    void TInputHandler::CursorWheelInputCallback(GLFWwindow *window, double XOffset, double YOffset)
    {
        TRenderer::CameraPos.z -= YOffset;
        if (DEBUG_MOUSE_WHEEL)
        {
            std::cout << glm::to_string(TRenderer::CameraPos) << std::endl;
        }
    }

    void TInputHandler::MouseInputCallback(GLFWwindow *window, int Button, int Action, int Mods)
    {
        if (Button == GLFW_MOUSE_BUTTON_RIGHT)
        {
            if (Action == GLFW_RELEASE)
            {
                TRenderer::RightMousePressed = false;

                double xPos, yPos;
                glfwGetCursorPos(window, &xPos, &yPos);
                TRenderer::PressedMousePos = {xPos, yPos};
            }
            else if (Action == GLFW_PRESS)
            {
                TRenderer::RightMousePressed = true;
            }
        }
    }

    void TInputHandler::MouseCursorMoveCallback(GLFWwindow *Window, double XPos, double YPos)
    {
        if (TRenderer::RightMousePressed)
        {
            auto pos = TVec2(XPos, YPos);
            const auto delta = (TRenderer::PressedMousePos - pos);

            TRenderer::CameraPos =
                glm::rotate(TRenderer::CameraPos, glm::length(delta) / TRenderer::MRSDivideFactor, TVec3(delta.y, delta.x, 0)); // inverted

            TRenderer::PressedMousePos = pos;
            if (DEBUG_MOUSE_POS)
            {
                std::cout << glm::to_string(delta) << std::endl;
            }
        }
    }
    void TInputHandler::KeyboardInputCallback(GLFWwindow *window, int key, int scancode, int action, int mods)
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

    void TInputHandler::KeyboardInputPressed(GLFWwindow *window, EKeys key, int scancode, int mods)
    {
        if (PressedKeys.contains(key))
        {
            auto state = PressedKeys[key];
            state.IsPressed = true;

            state.Callback.Execute(state.IsPressed);
        }
    }

    void TInputHandler::KeyboardInputReleased(GLFWwindow *window, EKeys key, int scancode, int mods)
    {
        if (PressedKeys.contains(key))
        {
            auto state = PressedKeys[key];
            state.IsPressed = false;

            state.Callback.Execute(state.IsPressed);
        }
    }
} // namespace RenderAPI
