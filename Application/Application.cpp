#define GLEW_STATIC

#include "glfw3.h"
#include "Application.hpp"

#include <string>
#include "Renderer.hpp"
#include <iostream>
#include <TestSimpleCube.hpp>
#include "TestTexture.hpp"

void Application::Start()
{
    
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE,GLFW_OPENGL_CORE_PROFILE);

    
    RenderAPI::Renderer renderer;

    GLFWwindow* window = renderer.Init("../../Externals/Shaders/SimpleCube.shader");
    while (!glfwWindowShouldClose(window))
        renderer.RenderTick();
    

    exit(EXIT_SUCCESS);
}
