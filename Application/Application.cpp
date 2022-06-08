#define GLEW_STATIC

#include "Application.hpp"
#include <string>
#include "Renderer.hpp"
#include <iostream>
#include <TestSimpleCube.hpp>
#include <Particle/TestParticles.hpp>
#include "TestTexture.hpp"
#include "SimpleBox.hpp"
#include "TestTube.hpp"


Application* Application::application;


void Application::Start(int argc, char **argv)
{

    RenderAPI::Renderer* renderer = RenderAPI::Renderer::getRenderer();
    renderer->GLFWInit();

    test::TestTube tube("E:/ProgrammingStuff/Projects/OpenGL/Externals/Shaders/SimpleCube.shader");
    renderer->addTest(&tube);

    renderer->GLFWRenderTickStart();
}
