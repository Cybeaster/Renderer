#define GLEW_STATIC

#include "Application.hpp"
#include <string>
#include "Renderer.hpp"
#include <iostream>
#include <TestSimpleCube.hpp>
#include <Particle/TestParticles.hpp>
#include "TestTexture.hpp"
#include "SimpleBox.hpp"

std::unique_ptr<Application> Application::application = nullptr;

void Application::Start(int argc, char **argv)
{
    const std::unique_ptr<RenderAPI::Renderer> renderer = RenderAPI::Renderer::getRenderer();

	if(!renderer) return;
    renderer->GLFWInit();

    // Add different tests or write your own.
    Test::TestParticles cube("D://Programs//ProgrammingStuff//OpenGL//Externals//Shaders//SimpleCube.shader");

    renderer->AddTest(&cube);
    renderer->GLFWRenderTickStart();
}
