#define GLEW_STATIC
#include "Application.hpp"

#include "Renderer.hpp"
#include "TestLightning.hpp"
#include "TestModelling.hpp"
#include "TestTexture.hpp"
#include "UnitTests/TestGroup.hpp"
#include "Window/GlfwWindow.hpp"

#include <string>

namespace RAPI
{

void OApplication::Start(int argc, char** argv)
{
	ParseInput(argc, argv);
	InitRenderer();
}

void OApplication::InitConsoleInput()
{
}

void OApplication::InitRenderer()
{
	CreateWindow();
	SetupInput();

	auto brickTexture = GetResourceDirectoryWith(TEXT("BrickWall.jpg"));
	auto earthTexture = GetResourceDirectoryWith(TEXT("TopographicalEarth.jpg"));
	auto shuttleModel = GetResourceDirectoryWith(TEXT("crawler.obj"));
	auto simpleCubeModel = GetResourceDirectoryWith(TEXT("Cube.obj"));

	auto renderer = ORenderer::Get();
	renderer->Init();
	renderer->SetInput(&InputHandler);

	auto simpleCube = Importer.BuildModelFromPath(simpleCubeModel, EModelType::OBJ);
	// OTestTexture textureTest(brickTexture, earthTexture, GetShaderLocalPathWith(BasicShader), renderer, simpleCube.get());

	OTestLightning lightning(brickTexture, earthTexture, GetShaderLocalPathWith(BasicPhongShading), renderer);
	renderer->AddTest(&lightning);

	while (!Window.get()->NeedClose())
	{
		Window->DrawStart();

		renderer->Tick(MakeRendererContext());

		Window->DrawEnd();
	}
}

void OApplication::ParseInput(int argc, char** argv)
{
	auto* arguments = new OString[argc];
	for (int it = 0; it < argc; it++)
	{
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
