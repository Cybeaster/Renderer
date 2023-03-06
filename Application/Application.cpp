#define GLEW_STATIC
#include "Application.hpp"

#include "Renderer.hpp"
#include "TestTexture.hpp"

#include <TestSimpleSolarSystem.hpp>
#include <iostream>
#include <string>

void OApplication::Start(int /*argc*/, char** /*argv*/)
{
	auto renderer = RenderAPI::ORenderer::Get();

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
	renderer->GLFWRenderTickStart();
}