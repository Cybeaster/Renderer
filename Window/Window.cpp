#include "Window.hpp"

#include "Application/Application.hpp"

namespace RAPI
{

void OWindow::OnMousePosition(double NewX, double NewY)
{
	MousePosition.x = static_cast<float>(NewX);
	MousePosition.y = static_cast<float>(NewY);
}

void OWindow::InitWindow()
{
	OApplication::GetApplication()->GetInputHandler()->AddRawMouseListener(this, &OWindow::OnMousePosition);
}

OVec2 OWindow::GetNDC() const
{
	OVec2 result;
	result.x = ((2 * MousePosition.x) / Width) - 1.f;
	result.y = 1.f - (2 * MousePosition.y) / Height;
	return result;
}
OVec2 OWindow::GetWidthHeight() const
{
	return { Width, Height };
}

} // namespace RAPI