#include <TestSimpleCube.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
namespace test
{
    
    void TestSimpleCube::OnRender(GLFWwindow* window,float currentTime)
    {
        GLCall(glClear(GL_DEPTH_BUFFER_BIT));
        GLCall(glClear(GL_COLOR_BUFFER_BIT));
        shader.Bind();

        glfwGetFramebufferSize(window,&width,&height);

        
        for (size_t i = 0; i < 24; i++)
        {
            float tf = currentTime + i;
        

            //calc animation:
            tMat = glm::translate(glm::mat4(1.0f),
            glm::vec3(sin(0.35f * tf) * 8.f,
            cos(0.52*tf) * 8.0f,
             sin(0.7f * tf) * 8.0f));

            rMat = glm::rotate(glm::mat4(1.0f),
            1.75f*currentTime,
            glm::vec3(0.0f,1.0f,0.0f));
            rMat = glm::rotate(rMat,1.75f*currentTime, glm::vec3(0.0f,0.0f,1.0f));
            rMat = glm::rotate(rMat,1.75f*currentTime, glm::vec3(1.0f,0.0f,0.0f));
            
        


            aspect = float(width) / float(height);
            
            pMat = glm::perspective(1.0472f,aspect,0.01f,1000.f);
            vMat = glm::translate(glm::mat4(1.0f),cameraPos * -1.f);
            mMat = glm::translate(glm::mat4(1.0f),cubePos);

            mMat = tMat * rMat;
            mvMat = vMat * mMat;

            shader.SetUnformMat4f("mv_Matrix",mvMat);
            shader.SetUnformMat4f("proj_Matrix",pMat);

            GLCall(glBindBuffer(GL_ARRAY_BUFFER,vbo[0]));
            GLCall(glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,0,0)); //says that data in the buffer are arrange in order of 3 cooridinates for a point;
            GLCall(glEnableVertexAttribArray(0));

            GLCall(glEnable(GL_DEPTH_TEST));
            GLCall(glDepthFunc(GL_LEQUAL));
            GLCall(glDrawArraysInstanced(GL_TRIANGLES,0,36,24));
        }
    }

    TestSimpleCube::TestSimpleCube(/* args */)
    {
        GLCall(glGenVertexArrays(1,vao));
        GLCall(glBindVertexArray(vao[0]));
        GLCall(glGenBuffers(2,vbo));

        GLCall(glBindBuffer(GL_ARRAY_BUFFER,vbo[0]));
        GLCall(glBufferData(GL_ARRAY_BUFFER,sizeof(vertexPositions),vertexPositions,GL_STATIC_DRAW));

    }
    
    TestSimpleCube::~TestSimpleCube()
    {

    }
}
