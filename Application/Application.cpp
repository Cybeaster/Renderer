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
    const auto path = GetShaderLocalPath();
    auto ResourceDir = GetResourceDirectory();
#ifdef DEBUG
    std::cout << path << std::endl;
    
#endif

    // Add different tests or write your own.
    Test::TestTexture test(ResourceDir.concat("Parrot.jpg"));
    renderer->AddTest(&test);
    renderer->GLFWRenderTickStart();
}