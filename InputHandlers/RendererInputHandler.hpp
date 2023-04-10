#pragma once

#include "KeyboardKeys.hpp"
#include "SmartPtr.hpp"
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

private:
	void OnWKeyToggled(bool Pressed);
	void OnSKeyToggled(bool Pressed);
	void OnDKeyToggled(bool Pressed);
	void OnAKeyToggled(bool Pressed);
	void OnEKeyToggled(bool Pressed);
	void OnQKeyToggled(bool Pressed);

	ORenderer* Owner;
};

} // namespace RAPI
