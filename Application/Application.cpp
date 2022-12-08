#define GLEW_STATIC

#include "Application.hpp"
#include <string>
#include <Windows.h>
#include "Renderer.hpp"
#include <iostream>
#include <filesystem>
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

    auto dirPath = std::filesystem::current_path();
    
    // for (auto dirIterator : std::filesystem::directory_iterator("/"))
    // {
    //     std::cout << dirIterator.path() << std::endl;
    // }
    //// Add different tests or write your own.
    // Test::TestParticles cube("D://Programs//ProgrammingStuff//OpenGL//Externals//Shaders//SimpleCube.shader");

    // renderer->AddTest(&cube);
    renderer->GLFWRenderTickStart();
}
