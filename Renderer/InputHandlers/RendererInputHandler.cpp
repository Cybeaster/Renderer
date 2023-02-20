#include "RendererInputHandler.hpp"

#include "Debug/Log.hpp"
#include "InputHandler.hpp"
#include "Renderer.hpp"

namespace RenderAPI
{
void ORendererInputHandler::OnWKeyToggled(bool Pressed)
{
	RAPI_LOG(Log, "On W KeyToggled called {Is pressed: %s}", Pressed ? "true" : "false");
}

void ORendererInputHandler::OnSKeyToggled(bool Pressed)
{
	RAPI_LOG(Log, "On S KeyToggled called {Is pressed: %s}", Pressed ? "true" : "false");
}

void ORendererInputHandler::OnDKeyToggled(bool Pressed)
{
	RAPI_LOG(Log, "On D KeyToggled called {Is pressed: %s}", Pressed ? "true" : "false");
}

void ORendererInputHandler::OnAKeyToggled(bool Pressed)
{
	RAPI_LOG(Log, "On A KeyToggled called {Is pressed: %s}", Pressed ? "true" : "false");
}

void ORendererInputHandler::BindKeys()
{
	Owner->AddRawListener(this, &ORendererInputHandler::OnWKeyToggled, EKeys::KEY_W);
	Owner->AddRawListener(this, &ORendererInputHandler::OnWKeyToggled, EKeys::KEY_W);
	Owner->AddRawListener(this, &ORendererInputHandler::OnAKeyToggled, EKeys::KEY_A);
	Owner->AddRawListener(this, &ORendererInputHandler::OnDKeyToggled, EKeys::KEY_D);
	Owner->AddRawListener(this, &ORendererInputHandler::OnSKeyToggled, EKeys::KEY_S);
}

} // namespace RenderAPI
