#define GLEW_STATIC
#include "Application.hpp"

#include "Renderer.hpp"
#include "TestTexture.hpp"
#include <TestSimpleSolarSystem.hpp>
#include <iostream>
#include <string>



void OApplication::Start(int /*argc*/, char** /*argv*/)
{
	auto renderer = RenderAPI::ORenderer::GetRenderer();

	if (!renderer)
		return;

	renderer->GLFWInit();
	const auto textureShaderPath = GetShaderLocalPathWith(SimpleTextureShader);
	const auto simpleCubeShader = GetShaderLocalPathWith(SimpleCubeShader);
	auto brickTexture = GetResourceDirectoryWith("BrickWall.jpg");
#ifndef NDEBUG

	std::cout << textureShaderPath << std::endl;

#endif

	// Add different tests or write your own.
	Test::OTestSimpleSolarSystem test(simpleCubeShader, renderer);
	Test::OTestTexture textureTest(brickTexture, textureShaderPath, renderer);

	// renderer->AddTest(&test);
	renderer->AddTest(&textureTest);
	renderer->GLFWRenderTickStart();
}