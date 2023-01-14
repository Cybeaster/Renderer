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
    const auto Shaderpath = GetShaderLocalPathWith(SimpleTextureShader);
    auto parrotPath = GetResourceDirectoryWith("BrickWall.jpg");
#ifdef DEBUG
    std::cout << Shaderpath << std::endl;

#endif

    // Add different tests or write your own.
    Test::TestTexture test(parrotPath, Shaderpath, renderer);
    renderer->AddTest(&test);
    renderer->GLFWRenderTickStart();
}