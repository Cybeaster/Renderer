#pragma once
#include "../Test/Test.hpp"
#include "Delegate.hpp"
#include "Hash.hpp"
#include "KeyboardKeys.hpp"
#include "RendererInputHandler.hpp"
#include "Types.hpp"
#include "Utils/Delegate/MulticastDelegate.hpp"

#include <Set.hpp>

namespace RenderAPI
{

struct SKeyState
{
	OMulticastDelegate<bool> Callback;
	bool IsPressed = false;
};

class OInputHandler
{
public:
	OInputHandler() = default;

	~OInputHandler() = default;

	static void KeyboardInputPressed(GLFWwindow* window, EKeys key, int scancode, int mods);
	static void KeyboardInputReleased(GLFWwindow* window, EKeys key, int scancode, int mods);

	static void KeyboardInputCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
	static void MouseCursorMoveCallback(GLFWwindow* Window, double XPos, double YPos);
	static void MouseInputCallback(GLFWwindow* window, int Button, int Action, int Mods);
	static void CursorWheelInputCallback(GLFWwindow* window, double XOffset, double YOffset);
	static void WindowReshapeCallback(GLFWwindow* window, int newHeight, int newWidth);

	void SetInput(GLFWwindow* Window);
	void Tick(float DeltaTime);

	template<typename ObjectType, typename... ArgTypes>
	void AddListener(ObjectType* Object, typename STMemberFunctionType<ObjectType, void, ArgTypes...>::Type Function, EKeys Key);

private:
	ORendererInputHandler RenderInputHandler{ this };

	static OTHashMap<EKeys, SKeyState> KeyMap;
};

template<typename ObjectType, typename... ArgTypes>
void OInputHandler::AddListener(ObjectType* Object, typename STMemberFunctionType<ObjectType, void, ArgTypes...>::Type /*Function*/, EKeys Key)
{
	if (Object != nullptr)
	{
	}
}
} // namespace RenderAPI