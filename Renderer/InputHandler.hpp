#pragma once
#include "../Test/Test.hpp"
#include "Types.hpp"
#include "Delegate.hpp"
#include "KeyboardKeys.hpp"
#include <Set.hpp>
namespace RenderAPI
{

    struct TKeyState
    {
        TTDelegate<bool> Callback;
        bool IsPressed = false;
    };

    class TInputHandler
    {
    public:
        TInputHandler()= default;
        
        ~TInputHandler();

        static void KeyboardInputPressed(GLFWwindow *window, EKeys key, int scancode, int mods);
        static void KeyboardInputReleased(GLFWwindow *window, EKeys key, int scancode, int mods);

        static void KeyboardInputCallback(GLFWwindow *window, int key, int scancode, int action, int mods);
        static void MouseCursorMoveCallback(GLFWwindow *Window, double XPos, double YPos);
        static void MouseInputCallback(GLFWwindow *window, int Button, int Action, int Mods);
        static void CursorWheelInputCallback(GLFWwindow *window, double XOffset, double YOffset);
        static void WindowReshapeCallback(GLFWwindow *window, const int newHeight, const int newWidth);

        void SetInput(GLFWwindow* Window);

        void Tick(float DeltaTime);
    private:
        static TTHashMap<EKeys,TKeyState> PressedKeys;
    };

}