#define GLEW_STATIC
#include "Application.hpp"
#include <string>
#include <Windows.h>
#include "Renderer.hpp"
#include <iostream>
#include <TestSimpleCube.hpp>
#include <Particle/TestParticles.hpp>
#include "TestTexture.hpp"
#include "SimpleBox.hpp"

void Application::Start(int argc, char **argv)
{
    auto renderer = RenderAPI::Renderer::GetRenderer();

    if (!renderer)
        return;

    renderer->GLFWInit();
    // Add different tests or write your own.
    Test::TestParticles cube(GetShaderLocalPath());

    // renderer->AddTest(&cube);
    renderer->GLFWRenderTickStart();
}