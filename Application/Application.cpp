#define GLEW_STATIC
#include "GL/glew.h"
#include <VertexArray.hpp>
#include <VertexBufferLayout.hpp>
#include "Application.hpp"
#include "GLFW/glfw3.h"
#include <string>
#include <iostream>
#include <VertexArray.hpp>
#include <VertexBuffer.hpp>
#include <Renderer.hpp>
#include <IndexBuffer.hpp>
#include <Shader.hpp>
#include <Texture.hpp>
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
    window = glfwCreateWindow(640, 480, "My new application", NULL, NULL);

    if (!window)
    {
        glfwTerminate();
    }

    /* Make the window's context current */
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);
    if(glewInit() != GLEW_OK)
        std::cout<<"Error with glewInit()"<<std::endl;
    

    std::cout<<glGetString(GL_VERSION)<<std::endl;
    {
        float positions[] = {
            -0.5f,-0.5, 0.0f,0.0f, //0
            0.5f,-0.5f, 1.0f,0.0,  // 1
            0.5f,0.5f,  1.0f,1.0f, //2
            -0.5f,0.5f, 0.0f,1.0f  //3
        };


        const uint32_t indices[] = {
            0, 1, 2, //first triangle
            2, 3 ,0
        };
        
        VertexArray va;
        VertexBuffer vb(positions,4 * 4  * sizeof(float));

        VertexBufferLayout layout;
        layout.Push<float>(2);
        layout.Push<float>(2);
        va.AddBuffer(vb,layout);
        
        IndexBuffer ib(indices,6);
        

        Shader shader("../../Externals/Shaders/Basic.shader");
        shader.Bind();
        shader.SetUniform4f("u_color",0.1f,0.3f,0.4f,1.f);

        Texture texture("../../Externals/Resources/Parrot.jpg");
        texture.Bind();
        shader.SetUniform1i("u_Texture",0);



        va.Unbind();
        vb.Unbind();
        ib.Unbind();
        shader.Unbind();

        Renderer renderer;

        float r = 0.f;
        float increment = 0.05;
        /* Loop until the user closes the window */
        while (!glfwWindowShouldClose(window))
        {
            renderer.Clear();
            
            shader.Bind();
            shader.SetUniform4f("u_color",r,0.3f,0.4f,1.f);

            va.Bind();
            ib.Bind();

            renderer.Draw(va,ib,shader);


            if(r > 1.f)
                increment = -0.05;
            else if (r < 0.0f)
                increment = 0.05;

            r += increment;

            /* Swap front and back buffers */
            glfwSwapBuffers(window);

            /* Poll for and process events */
            glfwPollEvents();
        }
    }
    glfwTerminate();

}
