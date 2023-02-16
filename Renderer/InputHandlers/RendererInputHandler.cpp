#include "RendererInputHandler.hpp"

#include "InputHandler.hpp"
#include "Renderer.hpp"

namespace RenderAPI
{
void ORendererInputHandler::OnWKeyToggled(bool Pressed)
{
}

void ORendererInputHandler::OnSKeyToggled(bool Pressed)
{
}

void ORendererInputHandler::OnDKeyToggled(bool Pressed)
{
}

void ORendererInputHandler::OnAKeyToggled(bool Pressed)
{
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
