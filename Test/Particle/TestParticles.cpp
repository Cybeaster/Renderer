#include "TestParticles.hpp"
#include "Renderer.hpp"


namespace test
{
    
    TestParticles::TestParticles(std::string shaderPath) : Test(shaderPath)
    {
        AddVertexArray();
        addParticle();
        AddBuffer(particles[0].getVertecies().begin()._Ptr,sizeof(float) * particles[0].getVertecies().size());

    }

    void TestParticles::addParticle()
    {
        glm::vec3 startPos(-40,10,0);
        glm::vec3 randPos{float( rand() % 5),float( rand() % 5), 0.f};
        particles.push_back(Particle(startPos + randPos,{},36,18,1));
    }
    TestParticles::~TestParticles()
    {

    }

    void TestParticles::OnUpdate(GLFWwindow* window,
        float deltaTime,
        float aspect,
        const glm::vec3& cameraPos,
        glm::mat4& pMat,
        glm::mat4& vMat)
    {
        Test::OnUpdate(window,deltaTime,aspect,cameraPos,pMat,vMat);

        spawnTick(deltaTime);
        for(auto& particle : particles)
        {
            particle.move({deltaTime, 0,0});
            
            particle.increaseSpeed(deltaTime * 10);
            getShader().SetUnformMat4f("mv_matrix",glm::translate(vMat,particle.getPosition()));

            EnableVertexArray(0);
            GLCall(glEnable(GL_DEPTH_TEST));
            GLCall(glFrontFace(GL_CCW));
            GLCall(glDepthFunc(GL_LEQUAL));
            GLCall(glDrawArrays(GL_TRIANGLES,0,particle.getVertecies().size()));
        }
    }

    void TestParticles::DrawParticle(const glm::vec3& Pos, float radius)
    {
        
    }

    void TestParticles::spawnTick(float DeltaTime)
    {
        if(particleSpawnTimer <= 0)
        {
            particleSpawnTimer = particleSpawnTime;
            addParticle();
        }
        else
            particleSpawnTimer -= DeltaTime;
    }
} // namespace test
