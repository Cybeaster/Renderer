
#pragma once
#include "Window.hpp"
#include "glfw3.h"

namespace RAPI
{
class OGLFWWindow : public OWindow
{
public:
	double GetDeltaTime() const override;

	~OGLFWWindow();

private:
	void CalcDeltaTime() override;
	void CalcAspect() override;
	bool NeedClose() override;
	void InitWindow() override;
	void DrawStart() override;
	void DrawEnd() override;

private:
	GLFWwindow* Window;
};

} // namespace RAPI
