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

	ORendererInputHandler(OVec3& Camera, OInputHandler* Handler)
	//: CameraRef(MakeShared(&Camera)), InputHandler(MakeShared(Handler))
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

	void SetHandler(OVec3* Camera) { CameraRef = MakeShared(Camera); }

private:
	OSharedPtr<OVec3> CameraRef;
	OSharedPtr<OInputHandler> InputHandler;
};

} // namespace RenderAPI
