#define GLEW_STATIC
#include "Application.hpp"

#include "Renderer.hpp"
#include "TestTexture.hpp"
#include "UnitTests/TestGroup.hpp"

#include <TestSimpleSolarSystem.hpp>
#include <iostream>
#include <string>

namespace RAPI
{

void OApplication::Start(int argc, char** argv)
{
	RUN_TEST(ParallelAlgos);

	ParseInput(argc, argv);
	StartProgram();
}

void OApplication::StartProgram()
{
	auto renderer = RAPI::ORenderer::Get();
	if (!renderer)
		return;

	renderer->GLFWInit();
	const auto textureShaderPath = GetShaderLocalPathWith(SimpleTextureShader);
	const auto simpleCubeShader = GetShaderLocalPathWith(SimpleCubeShader);
	auto brickTexture = GetResourceDirectoryWith(TEXT("BrickWall.jpg"));
	auto earthTexture = GetResourceDirectoryWith(TEXT("TopographicalEarth.jpg"));
#ifndef NDEBUG
	std::cout << textureShaderPath << std::endl;
#endif
	Test::OTestTexture textureTest(brickTexture, earthTexture, textureShaderPath, renderer);
	// renderer->AddTest(&test);
	renderer->AddTest(&textureTest);
	renderer->Tick();
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
	auto renderer = RAPI::ORenderer::Get();
	InputHandler.InitHandlerWith(renderer->GetWindowContext());
}
} // namespace RAPI
