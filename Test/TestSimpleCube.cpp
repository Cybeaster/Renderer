#include <TestSimpleCube.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include "GLFW/glfw3.h"

namespace test
{
    
    float TestSimpleCube::aspect{0};
    glm::mat4 TestSimpleCube::pMat{};
    glm::mat4 TestSimpleCube::vMat{};
    glm::mat4 TestSimpleCube::mMat{};
    glm::mat4 TestSimpleCube::mvMat{};
    glm::mat4 TestSimpleCube::tMat{};
    glm::mat4 TestSimpleCube::rMat{};
    
    void WindowReshapeCallback(GLFWwindow* window,int newHeight,int newWidth)
    {
        TestSimpleCube::aspect = (float)newWidth / (float)newHeight;
        glViewport(0,0,newWidth,newHeight);
        TestSimpleCube::pMat = glm::perspective(1.0472f,TestSimpleCube::aspect,0.1f,1000.f);
    }

    void TestSimpleCube::OnRender(GLFWwindow* window,float currentTime)
    {
        GLCall(glClear(GL_DEPTH_BUFFER_BIT));
        GLCall(glClear(GL_COLOR_BUFFER_BIT));
        GLCall(glEnable(GL_CULL_FACE));
        shader.Bind();

        glfwGetFramebufferSize(window,&width,&height);

        aspect = float(width) / float(height);

        pMat = glm::perspective(1.0472f,aspect,0.01f,1000.f);
        shader.SetUnformMat4f("proj_matrix",pMat);


        //pyr
        vMat = glm::translate(glm::mat4(1.0f),cameraPos * -1.f);
        mvStack.push(vMat);

        mvStack.push(mvStack.top());
        mvStack.top()*= glm::translate(glm::mat4(1.0f),glm::vec3(0.0,0.0,0.0));

        mvStack.push(mvStack.top());
        mvStack.top()*= glm::rotate(glm::mat4(1.0f),float(currentTime),glm::vec3(1.0,0.0,0.0));

        shader.SetUnformMat4f("mv_matrix",mvStack.top());
        

        GLCall(glBindBuffer(GL_ARRAY_BUFFER,vbo[1]));
        GLCall(glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,0,0)); //says that data in the buffer are arrange in order of 3 cooridinates for a point;
        GLCall(glEnableVertexAttribArray(0));

        GLCall(glEnable(GL_DEPTH_TEST));
        GLCall(glFrontFace(GL_CCW));
        GLCall(glDepthFunc(GL_LEQUAL));
        GLCall(glDrawArrays(GL_TRIANGLES,0,18));
        
        mvStack.pop();

        //pyr


        //cube

        mvStack.push(mvStack.top());
        mvStack.top() *= glm::translate(glm::mat4(1.0f),glm::vec3(sin(float(currentTime))* 4.0f,0.0f,cos(float(currentTime)*4.0)));

        mvStack.push(mvStack.top());
        mvStack.top()*= glm::rotate(glm::mat4(1.0f),float(currentTime),glm::vec3(0.0,1.0,0.0));


       
        shader.SetUnformMat4f("mv_matrix",mvStack.top());

        GLCall(glBindBuffer(GL_ARRAY_BUFFER,vbo[1]))
        GLCall(glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,0,0));
        GLCall(glEnableVertexAttribArray(0));
        
        GLCall(glEnable(GL_DEPTH_TEST));
        GLCall(glFrontFace(GL_CCW));
        GLCall(glDepthFunc(GL_LEQUAL));
        GLCall(glDrawArrays(GL_TRIANGLES,0,36));
        mvStack.pop();

        //smaller cube

        mvStack.push(mvStack.top());
        mvStack.top()*= glm::translate(glm::mat4(1.0f),glm::vec3(0.f,sin(currentTime)*2.0,cos(float(currentTime)*2.0)));
        mvStack.top()*= glm::rotate(glm::mat4(1.0f),float(currentTime),glm::vec3(0.0,0.0,1.0));
        mvStack.top()*= glm::scale(glm::mat4(1.0),glm::vec3(0.25f,0.25f,0.25f));

        shader.SetUnformMat4f("mv_matrix",mvStack.top());

        GLCall(glBindBuffer(GL_ARRAY_BUFFER,vbo[1]))
        GLCall(glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,0,0));
        GLCall(glEnableVertexAttribArray(0));
        
        GLCall(glEnable(GL_DEPTH_TEST));
        GLCall(glFrontFace(GL_CW));
        GLCall(glDepthFunc(GL_LEQUAL));
        GLCall(glDrawArrays(GL_TRIANGLES,0,36));

         mvStack.pop(); mvStack.pop(); mvStack.pop(); mvStack.pop();



    }

 
    TestSimpleCube::TestSimpleCube(GLFWwindow* window)
    {
        GLCall(glGenVertexArrays(1,vao));
        GLCall(glBindVertexArray(vao[0]));
        GLCall(glGenBuffers(2,vbo));

        GLCall(glBindBuffer(GL_ARRAY_BUFFER,vbo[0]));
        GLCall(glBufferData(GL_ARRAY_BUFFER,sizeof(vertexPositions),vertexPositions,GL_STATIC_DRAW));

        GLCall(glBindBuffer(GL_ARRAY_BUFFER,vbo[1]));
        GLCall(glBufferData(GL_ARRAY_BUFFER,sizeof(pyramidPositions),pyramidPositions,GL_STATIC_DRAW));

     
        glfwSetWindowSizeCallback(window,WindowReshapeCallback);
    }

    TestSimpleCube::~TestSimpleCube()
    {

    }

   
}
