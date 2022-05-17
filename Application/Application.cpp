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

    
    //Инициализируем класс, отвечающий за рендер.
    RenderAPI::Renderer renderer;
    GLFWwindow* window = renderer.Init();

    //Создаем тест,передавая путь к шейдеру.
    test::TestParticles particles("X:/ProgrammingStuff/Projects/OpenGL/Externals/Shaders/SimpleCube.shader");

    //Каждый добавленный тест будет отрисовывать свою сцену независимо от другого
    //Контекст у них будет один, тот, что создал RenderAPI::Renderer
    renderer.addTest(&particles);
    
    //Кадровый тик
    while (!glfwWindowShouldClose(window))
        renderer.renderTick();
    

    exit(EXIT_SUCCESS);
}
