#pragma once
#include "Delegate.hpp"
#include "KeyboardKeys.hpp"
#include "RendererInputHandler.hpp"
#include "SmartPtr.hpp"
#include "TypeTraits.hpp"
#include "Types.hpp"
#include "Utils/Delegate/MulticastDelegate.hpp"
#include "Utils/Types/HashMap/Hash.hpp"
#include "Utils/Types/Set.hpp"
#include "glfw3.h"

namespace RAPI
{

struct SKeyState
{
	OMulticastDelegate<bool> Callback;
	bool IsPressed = false;
};

class OInputHandler
{
	friend ORendererInputHandler;

public:
	OInputHandler() = default;
	~OInputHandler() = default;

	explicit OInputHandler(GLFWwindow* Window, bool Enable = true);

	void Tick(float DeltaTime);

	static void KeyboardInputPressed(GLFWwindow* window, EKeys key, int scancode, int mods);
	static void KeyboardInputReleased(GLFWwindow* window, EKeys key, int scancode, int mods);
	static void KeyboardInputCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
	static void MouseCursorMoveCallback(GLFWwindow* Window, double XPos, double YPos);
	static void MouseInputCallback(GLFWwindow* window, int Button, int Action, int Mods);
	static void CursorWheelInputCallback(GLFWwindow* window, double XOffset, double YOffset);
	static void WindowReshapeCallback(GLFWwindow* window, int newHeight, int newWidth);

	template<typename ObjectType, typename FunctionType, typename... ArgTypes>
	void AddRawListener(ObjectType* Object, FunctionType Function, EKeys Key)
	{
		if (Object != nullptr)
		{
			KeyMap[Key].Callback.AddRaw<ObjectType>(Object, Function);
		}
	}

	template<typename ObjectType, typename FunctionType, typename... ArgTypes>
	void AddSharedListener(OSharedPtr<ObjectType> Object, FunctionType Function, EKeys Key)
	{
		if (Object != nullptr)
		{
			KeyMap[Key].Callback.AddSP(Object, Function);
		}
	}

	void InitHandlerWith(GLFWwindow* Window);

private:
	void SetInput(GLFWwindow* Window);

	static OHashMap<EKeys, SKeyState> KeyMap;
};

} // namespace RAPI