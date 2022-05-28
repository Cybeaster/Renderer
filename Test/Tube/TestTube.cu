
#include "TestTube.hpp"
#include "Renderer.hpp"
#include "glm.hpp"
#include "StaticParticle.hpp"
#include "mainTest.cpp"
namespace test
{
    TestTube::TestTube(std::string shaderPath) : Test(shaderPath)
    {
        AddVertexArray();
        AddVertexArray();

        addParticle({-80,20,0},1,1,particles45StartVel);

        AddBuffer(particles[0].getVertecies().begin()._Ptr,sizeof(float) * particles[0].getVertecies().size());
        //AddBuffer(magnitTube.getVertices().begin()._Ptr,sizeof(float) * magnitTube.getVertices().size());
        
        
    }

    TestTube::~TestTube()
    {

    }

    void TestTube::OnUpdate(GLFWwindow* window,
        float deltaTime,
        float aspect,
        const glm::vec3& cameraPos,
        glm::mat4& pMat,
        glm::mat4& vMat)
    {
        Test::OnUpdate(window,deltaTime,aspect,cameraPos,pMat,vMat);

        particleSpawnTick(deltaTime);        
        drawParticles(deltaTime,vMat);
        drawTube();
    }
    void TestTube::addParticle(const glm::vec3& startPos,const float& radius,const float& charge,const glm::vec3& startVelocity)
    {
        glm::vec3 randPos{float( rand() % 15),float( rand() % 15), 0.f};
        particles.push_back(Particle(startPos + randPos,startVelocity,36,18,radius,charge));
    }

    void TestTube::drawParticles(float deltaTime,glm::mat4 vMat)
    {
        for(auto& particle : particles)
        {
            particle.updateColor();
            moveParticle(particle,deltaTime,vMat);
            drawParticlePath(particle.getPosition());
            glm::vec3 color = particle.getColor();
            getShader().SetUniform4f("additionalColor",color.x,color.y,color.y,1);
            
            glDrawParticle(particle);
        }

        for(auto& particle : pathParticles)
        {
            moveParticle(particle,deltaTime,vMat);
            getShader().SetUniform4f("additionalColor",1,1,1,1);
            glDrawParticle(particle);
        }
    }
    void TestTube::glDrawParticle(const Particle& particle)
    {
        EnableVertexArray(0);
        GLCall(glEnable(GL_DEPTH_TEST));
        GLCall(glFrontFace(GL_CCW));
        GLCall(glDepthFunc(GL_LEQUAL));
        GLCall(glDrawArrays(GL_TRIANGLES,0,particle.getVertecies().size()));
    }

    void TestTube::drawParticlePath(const glm::vec3& pos)
    {
        pathParticles.push_back(StaticParticle(pos));
    }

    void TestTube::drawTube()
    {
       drawPath();
    }

    void TestTube::particleSpawnTick(float DeltaTime)
    {
        if(particleSpawnTimer <= 0)
        {
            particleSpawnTimer = particleSpawnTime;
            float charge = rand() % 100 > 100 ? -1 : 1;
            const glm::vec3 velocity = rand() % 100 > 50 ? particles45StartVel : particlesNegative45StartVel;
            addParticle({-80,20,0},rand() % 3,charge,velocity);
        }
        else
            particleSpawnTimer -= DeltaTime;
    }

        void TestTube::moveParticle(Particle& particle,float deltaTime, glm::mat4 vMat)
        {
            glm::mat4 translation = glm::translate(vMat,particle.getPosition());
            glm::mat4 rotation = particle.rotate(deltaTime);
            getShader().SetUnformMat4f("mv_matrix",translation * rotation);
            particle.increaseRotSpeed(deltaTime * 10);
            particle.move();
        }
}