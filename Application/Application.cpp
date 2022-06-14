#define GLEW_STATIC

#include "Application.hpp"
#include <string>
#include "Renderer.hpp"
#include <iostream>
#include <TestSimpleCube.hpp>
#include <Particle/TestParticles.hpp>
#include "TestTexture.hpp"
#include "SimpleBox.hpp"


Application* Application::application;


void Application::Start(int argc, char **argv)
{

    RenderAPI::Renderer* renderer = RenderAPI::Renderer::getRenderer();
    renderer->GLFWInit();
    
    //Add different tests or write your own.
    test::TestParticles tube("E:/ProgrammingStuff/Projects/OpenGL/Externals/Shaders/SimpleCube.shader");
    renderer->addTest(&tube);

    renderer->GLFWRenderTickStart();
}
