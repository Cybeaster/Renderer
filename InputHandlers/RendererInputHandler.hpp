#pragma once

#include "InputHandler.hpp"
#include "SmartPtr.hpp"
#include "Utils/Types/Keys/KeyboardKeys.hpp"
#include "Utils/Types/Math.hpp"

namespace RAPI
{
class OInputHandler;
class ORenderer;

class ORendererInputHandler
{
public:
	ORendererInputHandler() = default;

	void BindKeys(OInputHandler* InputHandler);
	void SetRenderer(ORenderer* Renderer);

private:
	void OnWKeyToggled(EKeyState State);
	void OnSKeyToggled(EKeyState State);
	void OnDKeyToggled(EKeyState State);
	void OnAKeyToggled(EKeyState State);
	void OnEKeyToggled(EKeyState State);
	void OnQKeyToggled(EKeyState State);

	void OnMouseLeftToggled(EKeyState State);
	void OnMouseRightToggled(EKeyState State);
	void OnMouseWheelToggled(EKeyState State);

	void OnMouseMoved(double XCoord, double YCoord);

	ORenderer* Owner{ nullptr };
	OVec2 MousePosition{ 0.0, 0.0 };

	bool IsRightMousePressed = false;

	bool XRotNegative = true;
	bool YRotNegative = false;


};

} // namespace RAPI
