#define GLEW_STATIC
#include "Application.hpp"
#include <string>
#include <Windows.h>
#include "Renderer.hpp"
#include <iostream>
#include <TestSimpleCube.hpp>
#include <Particle/TestParticles.hpp>
#include "TestTexture.hpp"

void Application::Start(int argc, char **argv)
{
    auto renderer = RenderAPI::TRenderer::GetRenderer();

    if (!renderer)
        return;

    renderer->GLFWInit();
    const auto path = GetShaderLocalPath();

    // Add different tests or write your own.
    Test::TestParticles cube(path, renderer.get());

    // renderer->AddTest(&cube);
    renderer->GLFWRenderTickStart();
}