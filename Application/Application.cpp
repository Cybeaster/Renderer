#define GLEW_STATIC
#include "Application.hpp"
#include <string>
#include <Windows.h>
#include "Renderer.hpp"
#include <iostream>
#include <TestSimpleCube.hpp>
#include <Particle/TestParticles.hpp>
#include <TestSimpleSolarSystem.hpp>
#include "TestTexture.hpp"

#define DEBUG

void Application::Start(int argc, char **argv)
{
    auto renderer = RenderAPI::TRenderer::GetRenderer();

    if (!renderer)
        return;

    renderer->GLFWInit();
    const auto textureShaderPath = GetShaderLocalPathWith(SimpleTextureShader);
    const auto simpleCubeShader = GetShaderLocalPathWith(SimpleCubeShader);
    auto brickTexture = GetResourceDirectoryWith("BrickWall.jpg");
#ifdef DEBUG

    std::cout << textureShaderPath << std::endl;

#endif

    // Add different tests or write your own.
    Test::TestSimpleSolarSystem test(simpleCubeShader, renderer);
    Test::TestTexture textureTest(brickTexture, textureShaderPath, renderer);

    // renderer->AddTest(&test);
    renderer->AddTest(&textureTest);
    renderer->GLFWRenderTickStart();
}