#include "TestParticles.hpp"
#include "Renderer.hpp"
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include "glm.hpp"

// Helper functions and utilities to work with CUDA
#include <helper_functions.h>
#include <helper_cuda.h>
#include <device_launch_parameters.h>

__global__ void calcParticleVel(float* particlePos, float* inc, float* result,const float incMultiplier, bool UsePositiveDir)
{
    int it = threadIdx.x;
    float incResult;

    if(UsePositiveDir)
        incResult  = (particlePos[it] - inc[it]) * incMultiplier;
    else
        incResult = (inc[it] - particlePos[it]) * incMultiplier;

    result[it] = particlePos[it] + 1.f * (incResult - particlePos[it]);

}

void calcVelocity(glm::vec3& outParticlePos,const glm::vec3& inc,float incMultiplier, bool UsePositiveDir)
{

    float host_ParticlePos[3] = {outParticlePos.x,outParticlePos.y,outParticlePos.z};
    float host_Inc[3] = {inc.x,inc.y,inc.z};

    float* device_ParticlePos = nullptr;
    float* device_Inc = nullptr;
    float* device_Result = nullptr;

    size_t vecSize = 3 * sizeof(float);

    checkCudaErrors(cudaMalloc(&device_ParticlePos,vecSize));
    checkCudaErrors(cudaMalloc(&device_Inc,vecSize));
    checkCudaErrors(cudaMalloc(&device_Result,vecSize));

    checkCudaErrors(cudaMemcpy(device_ParticlePos,&host_ParticlePos,vecSize,cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(device_Inc,&host_Inc,vecSize,cudaMemcpyHostToDevice));

    
    // Create and start timer
    printf("Computing result using CUDA Kernel...\n");


    calcParticleVel<<<1,3>>>(
        device_ParticlePos,
        device_Inc,
        device_Result,
        incMultiplier,
        UsePositiveDir);

    cudaDeviceSynchronize();
    printf("done\n");

    checkCudaErrors(
        cudaMemcpy(&host_ParticlePos, device_Result, vecSize, cudaMemcpyDeviceToHost));


    for (size_t i = 0; i < 3; i++)
        printf("%f",host_ParticlePos[i]);
    
    checkCudaErrors(cudaFree(device_ParticlePos));
    checkCudaErrors(cudaFree(device_Inc));
    checkCudaErrors(cudaFree(device_Result));
}


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
    //Move all particles 
    void TestParticles::moveParticle(Particle& particle,float deltaTime, glm::mat4 vMat)
    {
        for(auto& field : electroFields) // Check distance to all fields to detect a collision.
        {
            float pointsDist = glm::distance(particle.getPosition(),field.particleField.getPosition());
            if(pointsDist <  field.radius) 
            {
                if(particle.getCharge() == field.particleField.getCharge()) // If the particle and field have the same charge - Move particle in opossite dir.
                {
                    glm::vec3 pos = particle.getPosition();
                    calcVelocity(pos,field.particleField.getPosition(),field.fieldStrenght / pointsDist,false); //Call to cuda func
                    particle.incVelocity(pos);
                }
                else
                {
                    glm::vec3 pos = particle.getPosition();
                    calcVelocity(pos,field.particleField.getPosition(),field.fieldStrenght / pointsDist,true); //Call to cuda func
                    particle.incVelocity(pos);
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
