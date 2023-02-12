#pragma once

#include "../../Utils/Types/Math.hpp"
#include "KeyboardKeys.hpp"
#include "SmartPtr.hpp"

namespace RenderAPI
{
class OInputHandler;

class ORendererInputHandler
{
public:
	void OnWKeyToggled(bool Pressed);
	void OnSKeyToggled(bool Pressed);
	void OnDKeyToggled(bool Pressed);
	void OnAKeyToggled(bool Pressed);

	explicit ORendererInputHandler(OInputHandler* Renderer)
	    : Owner(Renderer)
	{
		// InputHandler->AddListener<ORendererInputHandler, bool>(
		//     this, &ORendererInputHandler::OnWKeyToggled, EKeys::KEY_W);
		// InputHandler->AddListener<ORendererInputHandler, bool>(
		//     this, &ORendererInputHandler::OnAKeyToggled, EKeys::KEY_A);
		// InputHandler->AddListener<ORendererInputHandler, bool>(
		//     this, &ORendererInputHandler::OnDKeyToggled, EKeys::KEY_D);
		// InputHandler->AddListener<ORendererInputHandler, bool>(
		//     this, &ORendererInputHandler::OnSKeyToggled, EKeys::KEY_S);
	}

	ORendererInputHandler() = default;

	OInputHandler* GetHandlingOwner()
	{
		return Owner.get();
	}

private:
	OTSharedPtr<OInputHandler> Owner;
};

} // namespace RenderAPI
