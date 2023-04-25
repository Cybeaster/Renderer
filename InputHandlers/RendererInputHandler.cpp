#include "RendererInputHandler.hpp"

#include "Logging/Log.hpp"
#include "Renderer.hpp"

namespace RAPI
{

void ORendererInputHandler::OnWKeyToggled(EKeyState State)
{
	RAPI_LOG(Log, "On W KeyToggled called (Is pressed: {}), (Camera position is: {})", TO_STRING(State), TO_STRING(Owner->GetCameraPosition()));

	if (State == EKeyState::Pressed)
	{
		Owner->MoveCamera(ETranslateDirection::Forward);
	}
}

void ORendererInputHandler::OnSKeyToggled(EKeyState State)
{
	RAPI_LOG(Log, "On S KeyToggled called (Is pressed: {}), (Camera position is: {})", TO_STRING(State), TO_STRING(Owner->GetCameraPosition()));
	if (State == EKeyState::Pressed)
	{
		Owner->MoveCamera(ETranslateDirection::Backward);
	}
}

void ORendererInputHandler::OnDKeyToggled(EKeyState State)
{
	RAPI_LOG(Log, "On D KeyToggled called (Is pressed: {}), (Camera position is: {})", TO_STRING(State), TO_STRING(Owner->GetCameraPosition()));
	if (State == EKeyState::Pressed)
	{
		Owner->MoveCamera(ETranslateDirection::Right);
	}
}

void ORendererInputHandler::OnAKeyToggled(EKeyState State)
{
	RAPI_LOG(Log, "On A KeyToggled called (Is pressed: {}), (Camera position is: {})", TO_STRING(State), TO_STRING(Owner->GetCameraPosition()));
	if (State == EKeyState::Pressed)
	{
		Owner->MoveCamera(ETranslateDirection::Left);
	}
}

void ORendererInputHandler::OnEKeyToggled(EKeyState State)
{
	RAPI_LOG(Log, "On E KeyToggled called (Is pressed: {}), (Camera position is: {})", TO_STRING(State), TO_STRING(Owner->GetCameraPosition()));
	if (State == EKeyState::Pressed)
	{
		Owner->MoveCamera(ETranslateDirection::Up);
	}
}

void ORendererInputHandler::OnQKeyToggled(EKeyState State)
{
	RAPI_LOG(Log, "On Q KeyToggled called (Is pressed: {}), (Camera position is: {})", TO_STRING(State), TO_STRING(Owner->GetCameraPosition()));
	if (State == EKeyState::Pressed)
	{
		Owner->MoveCamera(ETranslateDirection::Down);
	}
}

void ORendererInputHandler::OnMouseWheelToggled(EKeyState State)
{
	RAPI_LOG(Log, "On Mouse Wheel KeyToggled called (Is pressed: {}), (Camera position is: {})", TO_STRING(State), TO_STRING(Owner->GetCameraPosition()));
	if (State == EKeyState::Pressed)
	{
	}
}

void ORendererInputHandler::OnMouseLeftToggled(EKeyState State)
{
	RAPI_LOG(Log, "On Mouse Left KeyToggled called (Is pressed: {}), (Camera position is: {})", TO_STRING(State), TO_STRING(Owner->GetCameraPosition()));
	if (State == EKeyState::Pressed)
	{
	}
}

void ORendererInputHandler::OnMouseRightToggled(EKeyState State)
{
	RAPI_LOG(Log, "On Mouse Right KeyToggled called (Is pressed: {}), (Camera position is: {})", TO_STRING(State), TO_STRING(Owner->GetCameraPosition()));
	if (State == EKeyState::Pressed)
	{
		IsRightMousePressed = true;
	}
	else
	{
		IsRightMousePressed = false;
	}
}

void ORendererInputHandler::BindKeys(OInputHandler* InputHandler)
{
	if (ENSURE(InputHandler))
	{
		InputHandler->AddRawKeyListener(this, &ORendererInputHandler::OnWKeyToggled, EKeys::KEY_W);
		InputHandler->AddRawKeyListener(this, &ORendererInputHandler::OnAKeyToggled, EKeys::KEY_A);
		InputHandler->AddRawKeyListener(this, &ORendererInputHandler::OnDKeyToggled, EKeys::KEY_D);
		InputHandler->AddRawKeyListener(this, &ORendererInputHandler::OnSKeyToggled, EKeys::KEY_S);
		InputHandler->AddRawKeyListener(this, &ORendererInputHandler::OnEKeyToggled, EKeys::KEY_E);
		InputHandler->AddRawKeyListener(this, &ORendererInputHandler::OnQKeyToggled, EKeys::KEY_Q);

		InputHandler->AddRawKeyListener(this, &ORendererInputHandler::OnMouseLeftToggled, EKeys::MouseLeft);
		InputHandler->AddRawKeyListener(this, &ORendererInputHandler::OnMouseRightToggled, EKeys::MouseRight);
		InputHandler->AddRawKeyListener(this, &ORendererInputHandler::OnMouseWheelToggled, EKeys::MouseWheel);

		InputHandler->AddRawMouseListener(this, &ORendererInputHandler::OnMouseMoved);
	}
}

void ORendererInputHandler::OnMouseMoved(double XCoord, double YCoord)
{
	if (IsRightMousePressed && MousePosition != OVec2(0, 0))
	{
		OVec2 delta{ MousePosition.x - XCoord, MousePosition.y - YCoord };

		if (XRotNegative)
		{
			delta.x *= -1;
		}

		if (YRotNegative)
		{
			delta.y *= -1;
		}

		Owner->RotateCamera(Move(delta));
	}

	MousePosition.x = static_cast<float>(XCoord);
	MousePosition.y = static_cast<float>(YCoord);

	RAPI_LOG(Log, "On Mouse Moved is called! Current mouse position is {}", TO_STRING(MousePosition));
}

void ORendererInputHandler::SetRenderer(ORenderer* Renderer)
{
	Owner = Renderer;
}

} // namespace RAPI
