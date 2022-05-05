#define GLEW_STATIC

#include "Application.hpp"
#include <string>
#include "Renderer.hpp"
#include <iostream>
#include <TestSimpleCube.hpp>
#include <Particle/TestParticles.hpp>
#include "TestTexture.hpp"
#include "SimpleBox.hpp"


void Application::Start()
{
    
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE,GLFW_OPENGL_CORE_PROFILE);

    
    RenderAPI::Renderer renderer;
    GLFWwindow* window = renderer.Init();

    test::TestSimpleCube cube(window,"X:/ProgrammingStuff/Projects/OpenGL/Externals/Shaders/SimpleCube.shader");
    test::TestParticles particles("X:/ProgrammingStuff/Projects/OpenGL/Externals/Shaders/SimpleCube.shader");
    test::TestSimpleBox box("X:/ProgrammingStuff/Projects/OpenGL/Externals/Shaders/simpleBox.shader");

    renderer.addTest(&cube);
    renderer.addTest(&particles);
    renderer.addTest(&box);
    
    while (!glfwWindowShouldClose(window))
        renderer.renderTick();
    

    exit(EXIT_SUCCESS);
}
