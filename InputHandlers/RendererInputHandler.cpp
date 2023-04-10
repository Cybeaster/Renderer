#include "RendererInputHandler.hpp"

#include "Logging/Log.hpp"
#include "Renderer.hpp"

namespace RAPI
{

void ORendererInputHandler::OnWKeyToggled(bool Pressed)
{
	RAPI_LOG(Log, "On W KeyToggled called (Is pressed: {}), (Camera position is: {})", TO_STRING(Pressed), TO_STRING(Owner->GetCameraPosition()));

	if (Pressed)
	{
		Owner->MoveCamera(OVec3(0, 0, -1));
	}
}

void ORendererInputHandler::OnSKeyToggled(bool Pressed)
{
	RAPI_LOG(Log, "On S KeyToggled called (Is pressed: {}), (Camera position is: {})", TO_STRING(Pressed), TO_STRING(Owner->GetCameraPosition()));
	if (Pressed)
	{
		Owner->MoveCamera(OVec3(0, 0, 1));
	}
}

void ORendererInputHandler::OnDKeyToggled(bool Pressed)
{
	RAPI_LOG(Log, "On D KeyToggled called (Is pressed: {}), (Camera position is: {})", TO_STRING(Pressed), TO_STRING(Owner->GetCameraPosition()));
	if (Pressed)
	{
		Owner->MoveCamera(OVec3(1, 0, 0));
	}
}

void ORendererInputHandler::OnAKeyToggled(bool Pressed)
{
	RAPI_LOG(Log, "On A KeyToggled called (Is pressed: {}), (Camera position is: {})", TO_STRING(Pressed), TO_STRING(Owner->GetCameraPosition()));
	if (Pressed)
	{
		Owner->MoveCamera(OVec3(-1, 0, 0));
	}
}

void ORendererInputHandler::OnEKeyToggled(bool Pressed)
{
	RAPI_LOG(Log, "On E KeyToggled called (Is pressed: {}), (Camera position is: {})", TO_STRING(Pressed), TO_STRING(Owner->GetCameraPosition()));
	if (Pressed)
	{
		Owner->MoveCamera(OVec3(0, 1, 0));
	}
}

void ORendererInputHandler::OnQKeyToggled(bool Pressed)
{
	RAPI_LOG(Log, "On Q KeyToggled called (Is pressed: {}), (Camera position is: {})", TO_STRING(Pressed), TO_STRING(Owner->GetCameraPosition()));
	if (Pressed)
	{
		Owner->MoveCamera(OVec3(0, -1, 0));
	}
}

void ORendererInputHandler::BindKeys(OInputHandler* InputHandler)
{
	if (ENSURE(InputHandler))
	{
		InputHandler->AddRawListener(this, &ORendererInputHandler::OnWKeyToggled, EKeys::KEY_W);
		InputHandler->AddRawListener(this, &ORendererInputHandler::OnWKeyToggled, EKeys::KEY_W);
		InputHandler->AddRawListener(this, &ORendererInputHandler::OnAKeyToggled, EKeys::KEY_A);
		InputHandler->AddRawListener(this, &ORendererInputHandler::OnDKeyToggled, EKeys::KEY_D);
		InputHandler->AddRawListener(this, &ORendererInputHandler::OnSKeyToggled, EKeys::KEY_S);
		InputHandler->AddRawListener(this, &ORendererInputHandler::OnEKeyToggled, EKeys::KEY_E);
		InputHandler->AddRawListener(this, &ORendererInputHandler::OnQKeyToggled, EKeys::KEY_Q);
	}
}

} // namespace RAPI
