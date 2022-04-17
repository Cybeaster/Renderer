#include "TestParticles.hpp"
#include "Renderer.hpp"


namespace test
{
    
    TestParticles::TestParticles(/* args */)
    {

    }
    
    TestParticles::~TestParticles()
    {

    }

    void TestParticles::OnUpdate(GLFWwindow* window, float deltaTime)
    {
        if(particleSpawnTimer <= 0)
            DrawParticle();
        else
            particleSpawnTimer-=deltaTime;

        for(auto& particle : particles)
        {
            GLCall(glEnable(GL_DEPTH_TEST));
            GLCall(glFrontFace(GL_CCW));
            GLCall(glDepthFunc(GL_LEQUAL));
            GLCall(glDrawArrays(GL_TRIANGLES,0,particle.GetNumOfVertices()));
        }
        
    }

    void TestParticles::DrawParticle(const glm::vec3& Pos, float radius)
    {
        
    }

} // namespace test
