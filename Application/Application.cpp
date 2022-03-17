#define GLEW_STATIC
#include "GL/glew.h"
#include "Application.hpp"
#include "GLFW/glfw3.h"
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>

#define ASSERT(x) if ((!x)) __debugbreak();

#define GLCall(x) GLClearError();\
    x;\
    ASSERT(GLLogCall(#x,__FILE__,__LINE__))


static void GLClearError()
{
    while(glGetError() != GL_NO_ERROR);
}

static bool GLLogCall(const char* func, const char* file, int line)
{
    while(GLenum error = glGetError())
    {
        std::cout<< "[Opengl Error] (" << std::hex<< error << ") :" << func <<'\t'<< line <<'\t'<< file<< std::endl;
        return false;
    }
    return true;
}


static ShaderSource ParseShader(const std::string& filePath)
{
    std::fstream stream(filePath);
    if(stream.is_open())
    {
        ShaderType currentType = ShaderType::NONE;
        std::string line;
        std::stringstream ss[2];

        while(getline(stream,line))
        {
            if(line.find("#shader") != std::string::npos)
            {
                if(line.find("vertex") != std::string::npos)
                {
                    currentType = ShaderType::VERTEX;
                }
                else if(line.find("fragment") != std::string::npos)
                {
                    currentType = ShaderType::FRAGMENT;
                }
            }
            else
            {
                ss[int(currentType)] << line << '\n';
            }
        }
        return {ss[0].str() , ss[1].str()};
    }

    return {};
}

static uint32_t CompileShader( uint32_t type,const std::string& source)
{
    uint32_t id = glCreateShader(type);
    const char* src = source.c_str();
    glShaderSource(id,1,&src,nullptr);
    glCompileShader(id);


    int32_t result;
    glGetShaderiv(id,GL_COMPILE_STATUS,&result);
    if(result == GL_FALSE)
    {
        int32_t lenght;
        glGetShaderiv(id,GL_INFO_LOG_LENGTH,&lenght);
        char* message = (char*)alloca(lenght * sizeof(char));
        glGetShaderInfoLog(id,lenght,&lenght,message);
        std::cout<<"Shaders isnt compiled"<<std::endl;
        std::cout<<message<<std::endl;
        glDeleteShader(id);
        return GL_FALSE;
    }
    return id;
}

static int CreateShader(const std::string& vertexShader, const std::string& fragmentShader)
{
    uint32_t program = glCreateProgram();
    uint32_t vs = CompileShader(GL_VERTEX_SHADER,vertexShader);
    uint32_t fs = CompileShader(GL_FRAGMENT_SHADER,fragmentShader);
    glAttachShader(program,vs);
    glAttachShader(program,fs);
    glLinkProgram(program);
    glValidateProgram(program);

    glDeleteShader(vs);
    glDeleteShader(fs);
    return program;
}


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

    float positions[12] = {
        -0.5f,-0.5, //0
         0.5f,-0.5f,// 1
         0.5f,0.5f, //2
        -0.5f,0.5f, //3
    };


    




    const uint32_t indices[] = {
        0, 1, 2, //first triangle
        2, 3 ,0
    };
    
    uint32_t vao;
    GLCall(glGenVertexArrays(1,&vao));
    GLCall(glBindVertexArray(vao));


    //Generates one buffer;
    uint32_t buffer;
    GLCall(glGenBuffers(1,&buffer));
    GLCall(glBufferData(GL_ARRAY_BUFFER,4 * 2 * sizeof(float),positions,GL_STATIC_DRAW));

    GLCall(glEnableVertexAttribArray(0));
    GLCall(glVertexAttribPointer(0,2,GL_FLOAT,GL_FALSE,sizeof(float)* 2,0));

    //Generates one buffer;
    uint32_t ibo;
    GLCall(glGenBuffers(1,&ibo));
    GLCall(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,ibo));
    GLCall(glBufferData(GL_ELEMENT_ARRAY_BUFFER,6 * sizeof(uint32_t),indices,GL_STATIC_DRAW));



    ShaderSource source = ParseShader("../../Externals/Shaders/Basic.shader");
    uint32_t shader = CreateShader(source.vertexShader,source.fragmentShader);

    GLCall(glUseProgram(shader));

    int32_t location = glGetUniformLocation(shader, "u_color");
    GLCall(glUniform4f(location,0.1f,0.3f,0.4f,1.f));


    GLCall(glBindVertexArray(0));
    GLCall(glUseProgram(0));
    GLCall(glBindBuffer(GL_ARRAY_BUFFER,0));
    GLCall(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,0));


    float r = 0.f;
    float increment = 0.05;
    /* Loop until the user closes the window */
    while (!glfwWindowShouldClose(window))
    {
        /* Render here */
        glClear(GL_COLOR_BUFFER_BIT);

        GLCall(glUseProgram(shader));
        GLCall(glUniform4f(location,0.1f,0.3f,0.4f,1.f));

        GLCall(glBindVertexArray(vao));
        GLCall(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,ibo));

        GLCall(glDrawElements(GL_TRIANGLES,6,GL_UNSIGNED_INT,nullptr));



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
    glDeleteProgram(shader);
    glfwTerminate();
}
