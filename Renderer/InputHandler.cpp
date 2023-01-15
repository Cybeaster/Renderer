#include "InputHandler.hpp"

namespace RenderAPI
{

    TInputHandler::~TInputHandler()
    {
    }

    void TInputHandler::Tick(float DeltaTime)
    {
    }

    void WindowReshapeCallback(GLFWwindow *window, const int newHeight, const int newWidth)
    {
        if (!window)
            return;
        glViewport(0, 0, newWidth, newHeight);

        Renderer->Aspect = static_cast<float>(newWidth / newHeight);
        Renderer->ScreenWidth = newWidth;
        Renderer->ScreenHeight = newHeight;
        Renderer->PMat = glm::perspective( Renderer->Fovy,  Renderer->Aspect, 0.1f, 1000.f);
    }

    void CursorWheelInputCallback(GLFWwindow *window, double XOffset, double YOffset)
    {
         Renderer->CameraPos.z -= YOffset;
        if (DEBUG_MOUSE_WHEEL)
        {
            std::cout << glm::to_string( Renderer->CameraPos) << std::endl;
        }
    }

    void MouseInputCallback(GLFWwindow *window, int Button, int Action, int Mods)
    {
        if (Button == GLFW_MOUSE_BUTTON_RIGHT)
        {
            if (Action == GLFW_RELEASE)
            {
                 Renderer->RightMousePressed = false;

                double xPos, yPos;
                glfwGetCursorPos(window, &xPos, &yPos);
                 Renderer->PressedMousePos = {xPos, yPos};
            }
            else if (Action == GLFW_PRESS)
            {
                 Renderer->RightMousePressed = true;
            }
        }
    }

    void MouseCursorMoveCallback(GLFWwindow *Window, double XPos, double YPos)
    {
        if ( Renderer->RightMousePressed)
        {
            auto pos = TVec2(XPos, YPos);
            const auto delta = ( Renderer->PressedMousePos - pos);

             Renderer->CameraPos =
                glm::rotate( Renderer->CameraPos, glm::length(delta) /  Renderer->MRSDivideFactor, TVec3(delta.y, delta.x, 0)); // inverted

             Renderer->PressedMousePos = pos;
            if (DEBUG_MOUSE_POS)
            {
                std::cout << glm::to_string(delta) << std::endl;
            }
        }
    }
    void KeyboardInputCallback(GLFWwindow *window, int key, int scancode, int action, int mods)
    {
        if (action == GLFW_PRESS)
        {
            KeyboardInputPressed(window, key, scancode, mods);
        }
        else
        {
            KeyboardInputReleased(window, key, scancode, mods);
        }
    }

    void KeyboardInputPressed(GLFWwindow *window, EKeys key, int scancode, int mods)
    {
        switch (key)
        {
        case Ekeys::KEY_W:
        {
            
            break;
        }
        case Ekeys::KEY_D:
        {

            break;
        }
        case Ekeys::KEY_S:
        {

            break;
        }
        case Ekeys::KEY_A:
        {

            break;
        }
        }
    }

    void KeyboardInputReleased(GLFWwindow *window, EKeys key, int scancode, int mods)
    {
    }
} // namespace RenderAPI
