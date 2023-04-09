#pragma once

#include "../../Utils/Types/Math.hpp"
#include "KeyboardKeys.hpp"
#include "SmartPtr.hpp"

namespace RAPI
{
class OInputHandler;
class ORenderer;
class ORendererInputHandler
{
public:
	ORendererInputHandler() = default;
	explicit ORendererInputHandler(OInputHandler* OwningHandler)
	    : Owner(OwningHandler)
	{
	}

	void BindKeys();

	OInputHandler* GetHandlingOwner()
	{
		return Owner;
	}

	void OnWKeyToggled(bool Pressed);
	void OnSKeyToggled(bool Pressed);
	void OnDKeyToggled(bool Pressed);
	void OnAKeyToggled(bool Pressed);
	void OnEKeyToggled(bool Pressed);
	void OnQKeyToggled(bool Pressed);

private:
	OInputHandler* Owner; // should be shared ref
};

} // namespace RAPI
