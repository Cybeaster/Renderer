#define GLEW_STATIC
#include "Application.hpp"

#include "Renderer.hpp"
#include "TestTexture.hpp"
#include "UnitTests/TestGroup.hpp"
#include "Window/GlfwWindow.hpp"

#include <TestSimpleSolarSystem.hpp>
#include <iostream>
#include <memory>
#include <string>

namespace RAPI
{

void OApplication::Start(int argc, char** argv)
{
	RUN_TEST(ParallelAlgos);

	ParseInput(argc, argv);
	InitRenderer();
	NamedThreadPool.AddTaskToThread(EThreadID::RenderThread, [this]() {});

	NamedThreadPool.AddTaskToThread(EThreadID::InputThread, [this]()
	                                { InitConsoleInput(); });
}

void OApplication::InitConsoleInput()
{
}

void OApplication::InitRenderer()
{
	CreateWindow();
	SetupInput();

	const auto textureShaderPath = GetShaderLocalPathWith(SimpleTextureShader);
	const auto simpleCubeShader = GetShaderLocalPathWith(SimpleCubeShader);

	auto brickTexture = GetResourceDirectoryWith(TEXT("BrickWall.jpg"));
	auto earthTexture = GetResourceDirectoryWith(TEXT("TopographicalEarth.jpg"));

	auto renderer = ORenderer::Get();

	renderer->Init();
	renderer->SetInput(&InputHandler);

	while (!Window.get()->NeedClose())
	{
		Window->DrawStart();

		renderer->Tick(MakeRendererContext());

		Window->DrawEnd();
	}
}

void OApplication::ParseInput(int argc, char** argv)
{
	OString arguments[argc];
	for (int it = 0; it < argc; it++)
	{
	}

	for (auto str : arguments)
	{
		RAPI_LOG(Log, "Command: {} is parsed", str);
		if (str == "StartTests")
		{
		}
	}
}
void OApplication::SetupInput()
{
	InputHandler.InitHandlerWith(dynamic_cast<OGLFWWindow*>(Window.get())->GetWindow());
}
void OApplication::CreateWindow()
{
	Window = OUniquePtr<OGLFWWindow>(new OGLFWWindow());
	Window.get()->InitWindow();
}

SRenderContext OApplication::MakeRendererContext() const
{
	auto* window = Window.get();

	SRenderContext context;
	context.AspectRatio = static_cast<float>(window->GetAspectRation());
	context.DeltaTime = static_cast<float>(window->GetDeltaTime());

	return context;
}

OApplication* OApplication::GetApplication()
{
	if (!Application)
	{
		Application = (new OApplication());
		return Application;
	}

	return Application;
}

OString OApplication::GetShaderLocalPathWith(const SShaderName& Name)
{
	return RootDirPath.string() + ShadersDir.string() + Name.Name;
}

OString OApplication::GetResourceDirectoryWith(const OPath& Path)
{
	return RootDirPath.string() + ResourceDirectory.string() + Path.string();
}

} // namespace RAPI
