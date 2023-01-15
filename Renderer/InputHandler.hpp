#pragma once
#include "../Test/Test.hpp"
#include "Types.hpp"
#include "KeyboardKeys.hpp"
#include <Set.hpp>
namespace RenderAPI
{
    class TInputHandler
    {
    public:
        TInputHandler(TTSharedPtr<TRenderer> RendererArg) : Renderer{RendererArg}
        {}
        
        ~TInputHandler();

        static void KeyboardInputPressed(GLFWwindow *window, EKeys key, int scancode, int mods);
        static void KeyboardInputReleased(GLFWwindow *window, EKeys key, int scancode, int mods);

        static void KeyboardInputCallback(GLFWwindow *window, int key, int scancode, int action, int mods);
        static void MouseCursorMoveCallback(GLFWwindow *Window, double XPos, double YPos);
        static void MouseInputCallback(GLFWwindow *window, int Button, int Action, int Mods);
        static void CursorWheelInputCallback(GLFWwindow *window, double XOffset, double YOffset);
        static void WindowReshapeCallback(GLFWwindow *window, const int newHeight, const int newWidth);

        void Tick(float DeltaTime);
    private:
        TTSet<Ekeys> PressedKeys;
        TTSharedPtr<TRenderer> Renderer;
    };

}