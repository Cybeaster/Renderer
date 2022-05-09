#include "TestParticles.hpp"
#include "Renderer.hpp"


namespace test
{
    
    TestParticles::TestParticles(std::string shaderPath) : Test(shaderPath)
    {
        AddVertexArray();
        addParticle();
        AddBuffer(particles[0].getVertecies().begin()._Ptr,sizeof(float) * particles[0].getVertecies().size());
        addField({75.f,-10.f,-1.f},0.5,{1.f,0.f,0.f});
        addField({75.f,20.f,-1.f},0.5,{1.f,0.f,0.f});
        addField({25.f,-10.f,-1.f},0.5,{1.f,0.f,0.f});
        addField({25.f,20.f,-1.f},0.5,{1.f,0.f,0.f});
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

        particleSpawnTick(deltaTime);
        //fieldSpawnTick(deltaTime);
        
        drawParticles(deltaTime,vMat);
        drawFields(deltaTime,vMat);
        
        
    }

    void TestParticles::moveParticle(Particle& particle,float deltaTime, glm::mat4 vMat)
    {
        for(auto& field : electroFields)
        {
            float pointsDist = glm::distance(particle.getPosition(),field.particleField.getPosition());
            if(pointsDist <  field.radius)
            {
                if(particle.getCharge() == field.particleField.getCharge())
                {
                    glm::vec3 inc = (particle.getPosition() - field.particleField.getPosition()) * field.fieldStrenght / pointsDist;
                    glm::vec3 res = glm::mix(particle.getPosition(),inc,1.f);
                    particle.incVelocity(res);
                }
                else
                {
                    glm::vec3 inc = (field.particleField.getPosition() - particle.getPosition()) * field.fieldStrenght / pointsDist;
                    glm::vec3 res = glm::mix(particle.getPosition(),inc,1.f);
                    particle.incVelocity(res);
                }
            }
        }
        
        glm::mat4 translation = glm::translate(vMat,particle.getPosition());
        glm::mat4 rotation = particle.rotate(deltaTime);

        particle.increaseRotSpeed(deltaTime * 10);
        particle.move();
        
        getShader().SetUnformMat4f("mv_matrix",translation * rotation);
    }

    void TestParticles::drawParticles(float deltaTime,glm::mat4 vMat)
    {
        for(auto& particle : particles)
        {
            moveParticle(particle,deltaTime,vMat);
            getShader().SetUniform4f("additionalColor",1,1,1,1);

            EnableVertexArray(0);
            GLCall(glEnable(GL_DEPTH_TEST));
            GLCall(glFrontFace(GL_CCW));
            GLCall(glDepthFunc(GL_LEQUAL));
            GLCall(glDrawArrays(GL_TRIANGLES,0,particle.getVertecies().size()));
        }
    }

    void TestParticles::particleSpawnTick(float DeltaTime)
    {
        if(particleSpawnTimer <= 0)
        {
            particleSpawnTimer = particleSpawnTime;
            addParticle();
        }
        else
            particleSpawnTimer -= DeltaTime;
    }

    void TestParticles::addField(const glm::vec3& pos,const float& charge,const glm::vec3& chargeVec)
    {
        Particle particle(pos,{},76,32,1,-1);
        ElectroField field(35,charge,particle);
        electroFields.push_back(field);
    }

    void TestParticles::fieldSpawnTick(float DeltaTime)
    {
        if(fieldSpawnTimer <= 0)
        {
            fieldSpawnTimer = fieldSpawnTime;
            addField({25.f,-10.f,-1.f},10,{1.f,0.f,0.f});
        }
        else
            fieldSpawnTimer -= DeltaTime;
    }

    void TestParticles::addParticle()
    {
        glm::vec3 startPos(-80,20,0);
        glm::vec3 randPos{float( rand() % 5),float( rand() % 5), 0.f};
        particles.push_back(Particle(startPos + randPos,particlesStartVel,36,18,1,1));
    }

    void TestParticles::drawFields(float deltaTime,glm::mat4 vMat)
    {
        for(auto& field : electroFields)
        {
            glm::mat4 rot = field.particleField.rotate(deltaTime);
            glm::mat4 translation = glm::translate(vMat,field.particleField.getPosition());

            field.particleField.increaseRotSpeed(10);

            getShader().SetUnformMat4f("mv_matrix",translation * rot);
            getShader().SetUniform4f("additionalColor",1,0,0,1);

            EnableVertexArray(0);
            GLCall(glEnable(GL_DEPTH_TEST));
            GLCall(glFrontFace(GL_CCW));
            GLCall(glDepthFunc(GL_LEQUAL));
            GLCall(glDrawArrays(GL_TRIANGLES,0,field.particleField.getVertecies().size()));
        }
    
    }


} // namespace test
