#pragma once
#include "../Test/Test.hpp"
#include "Delegate.hpp"
#include "Hash.hpp"
#include "KeyboardKeys.hpp"
#include "RendererInputHandler.hpp"
#include "SmartPtr.hpp"
#include "TypeTraits.hpp"
#include "Types.hpp"
#include "Utils/Delegate/MulticastDelegate.hpp"
#include "glfw3.h"

#include <Set.hpp>

namespace RenderAPI
{

struct SKeyState
{
	OTMulticastDelegate<bool> Callback;
	bool IsPressed = false;
};

class OIInputHandler
{
	NODISCARD virtual ORenderer* GetRenderer() const = 0;
};

class ORenderer;
class OInputHandler : public OIInputHandler
{
	friend ORendererInputHandler;

public:
	OInputHandler() = default;
	~OInputHandler() = default;

	explicit OInputHandler(ORenderer* OwningRender, bool Enable = true);

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
	void AddSharedListener(OTSharedPtr<ObjectType> Object, FunctionType Function, EKeys Key)
	{
		if (Object != nullptr)
		{
			KeyMap[Key].Callback.AddSP(Object, Function);
		}
	}

	NODISCARD ORenderer* GetRenderer() const override
	{
		return Renderer;
	}

	void InitHandlerWith(ORenderer* Owner);

protected:
	void MoveCamera(const OVec3& Offset);

private:
	void SetInput(GLFWwindow* Window);

	ORenderer* Renderer;
	ORendererInputHandler RenderInputHandler{ this };

	static OTHashMap<EKeys, SKeyState> KeyMap;
};

} // namespace RenderAPI