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
#include <glm/glm.hpp>
#include <Texture.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <imgui.h>
#include <imgui_impl_glfw_gl3.h>

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
    window = glfwCreateWindow(960, 960, "My new application", NULL, NULL);

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
            -50.f,-50.f, 0.0f,0.0f, //0
            50, -50.f, 1.0f,0.0,  //1
            50.f,50.f,  1.0f,1.0f, //2
            -50.f,50.f, 0.0f,1.0f  //3
        };


        const uint32_t indices[] = {
            0, 1, 2, //first triangle
            2, 3 ,0
        };
        
        GLCall(glEnable(GL_BLEND));
        GLCall(glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA));

        VertexArray va;
        VertexBuffer vb(positions,4 * 4  * sizeof(float));

        VertexBufferLayout layout;
        layout.Push<float>(2);
        layout.Push<float>(2);
        va.AddBuffer(vb,layout);
        
        IndexBuffer ib(indices,6);
        
        glm::mat4 proj = glm::ortho(0.f,960.f,0.f,960.f,-1.0f,1.0f);
        glm::mat4 view = glm::translate(glm::mat4(1.0f),glm::vec3(0,0,0));

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

        glm::vec3 translation(200.f,200.f,0.f);
        glm::vec3 translationA(400.f,200.f,0);
        glm::vec3 translationB(600.f,200.f,0);



        ImGui::CreateContext();
        ImGui_ImplGlfwGL3_Init(window,true);
        ImGui::StyleColorsDark();



        /* Loop until the user closes the window */
        while (!glfwWindowShouldClose(window))
        {
            renderer.Clear();
            
            ImGui_ImplGlfwGL3_NewFrame();


            glm::mat4 modelA = glm::translate(glm::mat4(1.f),translationA);
            glm::mat4 modelB = glm::translate(glm::mat4(1.f),translationB);
            glm::mat4 mvp; 

            shader.SetUniform4f("u_color",r,0.3f,0.4f,1.f);

            shader.Bind();
            {
                mvp = proj * view * modelA;
                shader.SetUnformMat4f("u_MVP",mvp);
                renderer.Draw(va,ib,shader);
            }

            {
                mvp = proj * view * modelB;
                shader.SetUnformMat4f("u_MVP",mvp);
                renderer.Draw(va,ib,shader);
            }


            if(r > 1.f)
                increment = -0.05;
            else if (r < 0.0f)
                increment = 0.05;

            r += increment;

                      
            ImGui::SliderFloat3("TranslationA", &translationA.x, 0.0f, 960.0f);        
            ImGui::SliderFloat3("TranslationB", &translationB.x, 0.0f, 960.0f);        

            ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
        



            ImGui::Render();
            ImGui_ImplGlfwGL3_RenderDrawData(ImGui::GetDrawData());

            /* Swap front and back buffers */
            glfwSwapBuffers(window);

            /* Poll for and process events */
            glfwPollEvents();
        }
    }
    ImGui_ImplGlfwGL3_Shutdown();
    ImGui::DestroyContext();
    glfwTerminate();

}
