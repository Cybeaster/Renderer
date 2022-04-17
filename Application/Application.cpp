#define GLEW_STATIC
#include "GL/glew.h"
#include "Application.hpp"
#include "GLFW/glfw3.h"
#include <string>
#include <iostream>
#include <TestSimpleCube.hpp>


void Application::Start()
{
    GLFWwindow* window;
    /* Initialize the library */
    if (!glfwInit())
    {}

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE,GLFW_OPENGL_CORE_PROFILE);

    /* Create a windowed mode window and its OpenGL context */
    window = glfwCreateWindow(1920, 1080, "My new application", NULL, NULL);

    if (!window)
        glfwTerminate();
    

    /* Make the window's context current */
    glfwMakeContextCurrent(window);

    glfwSwapInterval(1);

    if(glewInit() != GLEW_OK)
        std::cout<<"Error with glewInit()"<<std::endl;
    
    GLCall(glEnable(GL_BLEND));
    GLCall(glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA));

    test::TestSimpleCube simpleCube(window);
        
    while (!glfwWindowShouldClose(window))
    {
        simpleCube.OnRender(window,glfwGetTime());

        /* Swap front and back buffers */
        glfwSwapBuffers(window);
        glfwPollEvents();
    
    }

    glfwDestroyWindow(window);
    glfwTerminate();
    exit(EXIT_SUCCESS);
}
