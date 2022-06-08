
#include "TestTube.hpp"
#include "Renderer.hpp"
#include "glm.hpp"
#include "StaticParticle.hpp"
#include "ElectroField.hpp"
#include <iostream>
// Helper functions and utilities to work with CUDA
#include <helper_functions.h>
#include <helper_cuda.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>


#define DRAW_PARTICLE_PATH false

__global__ void calcCudaTranslation(float* vMat, float* position, float* result)
{
    int xIt = blockIdx.x * blockDim.x + threadIdx.x;

    if(xIt < 4)
    {
        for(int i = 0; i < 4; i++)
        {
            if(i == 3)
                result[xIt] += vMat[i * 4 + xIt];
            else
                result[xIt] += vMat[i * 4 + xIt] * position[i];
        }
    }
}


glm::mat4 calcTranslation(const glm::mat4& vMat, const glm::vec3 position)
{
    float host_vMat[16] = 
        {vMat[0][0],vMat[0][1],vMat[0][2],vMat[0][3],
        vMat[1][0],vMat[1][1],vMat[1][2],vMat[1][3],
        vMat[2][0],vMat[2][1],vMat[2][2],vMat[2][3],
        vMat[3][0],vMat[3][1],vMat[3][2],vMat[3][3]};

    
    float host_Position[3] =
    {
        position.x,position.y,position.z
    };

    float host_result[4];

    float* device_vMat;
    float* device_position;
    float* device_Result;

    size_t matSize = 16 * sizeof(float);
    size_t vecSize = 3 * sizeof(float);

    checkCudaErrors(cudaMalloc((void**)&device_vMat,matSize));
    checkCudaErrors(cudaMalloc((void**)&device_position,vecSize));
    checkCudaErrors(cudaMalloc((void**)&device_Result,vecSize));

    checkCudaErrors(cudaMemcpy(device_vMat,host_vMat,matSize,cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(device_position,host_Position,vecSize,cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(4, 4);
    dim3 blocksPerGrid(1,1);

    calcCudaTranslation<<<blocksPerGrid,threadsPerBlock>>>(device_vMat,device_position,device_Result);

    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(
        cudaMemcpy(host_result, device_Result, vecSize, cudaMemcpyDeviceToHost));

    glm::mat4 result(vMat);
    result[3] = {host_result[0],host_result[1],host_result[2],1.f};

    checkCudaErrors(cudaFree(device_vMat));
    checkCudaErrors(cudaFree(device_position));
    checkCudaErrors(cudaFree(device_Result));
    
    return result;
}

//Умножает вектор на скаляр
__global__ void CUDAmultiplyVecByScalar(float* vec, float* result, float scalar)
{
    int it = threadIdx.x;
    if(it < 3)
    {
        result[it] = vec[it] * scalar;
           __syncthreads();
    }
}
//host функция умножения вектора на скаляр
glm::vec3 multiplyVecByScalar(const glm::vec3& particlePos,const int32_t scalar)
{
    float* host_ParticlePos = new float[3];

    host_ParticlePos[0] = particlePos.x;
    host_ParticlePos[1] = particlePos.y;
    host_ParticlePos[2] = particlePos.z;

    float* host_Inc = new float[3];
    float* host_result = new float[3];

    float* device_ParticlePos = nullptr;
    float* device_Result = nullptr;

    size_t vecSize = 3 * sizeof(float);

    checkCudaErrors(cudaMalloc((void**)&device_ParticlePos,vecSize));
    checkCudaErrors(cudaMalloc((void**)&device_Result,vecSize));
    checkCudaErrors(cudaMemcpy(device_ParticlePos,host_ParticlePos,vecSize,cudaMemcpyHostToDevice));

    

    CUDAmultiplyVecByScalar<<<1, 3>>>(
        device_ParticlePos,
        device_Result,
        scalar);

    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(
        cudaMemcpy(host_result, device_Result, vecSize, cudaMemcpyDeviceToHost));

    glm::vec3 result{host_result[0],host_result[1],host_result[2]};

    checkCudaErrors(cudaFree(device_ParticlePos));
    checkCudaErrors(cudaFree(device_Result));

    delete[] host_ParticlePos;
    delete[] host_Inc;
    delete[] host_result;

    return result;
}

namespace test
{

    TestTube::TestTube(std::string shaderPath) : Test(shaderPath)
    {
        Init(shaderPath);
    }
        //Строит овал базируясь на параметрах
        std::vector<float> buildCircle(float radius,float offsetX, float offsetY,float angleDiv,float step )
        {
            const float PI = acos(-1);
            
            std::vector<float> vertices;
            
            //при radiuxX/Y = radius будет обычный круг
            float radiusX = 1 * radius;
            float radiusY = 1 * radius;

            //изначально PI/2 - четверть круга
            for(float tempAngle = 0; tempAngle <= PI / angleDiv ;tempAngle += step )
            {
                vertices.push_back(radiusX * sin(tempAngle) + offsetX);
                vertices.push_back(radiusY * cos(tempAngle) + offsetY);
                vertices.push_back(0.f);
            }
            return vertices;
        }

    /**
     * @brief Строит магнитные поля базируясь на уже нарисованных points
     * @param points точки, по которым будут строиться поля
     * @param magintPointSize радиус каждой точки
     */
    std::vector<ElectroField> getMagnitFields(std::vector<float> points,float magintPointSize)
    {
        std::vector<ElectroField> fields;
        for (size_t i = 0; i < points.size(); i += 3)
        {
            //берем из вектора всех точек, которые будем отрисовывать, 3 основных
            //Каждые 3 точки в векторе - позиции x,y,z
            
            glm::vec3 fieldPos(points[i],points[i+1],points[i+2]);
            //каждая точка хранит свою позицию
            //для определения вектора силы, которая будет толкать Particle, необходимо знать точку A начальную, и B конечную
            //разность векторов позиций этих точек даст вектор направления, в котором нужно будт двигать точку
            //У самой первой точки такого вектора силы не будет
            if(fields.size() > 0)
            {

                ElectroField& previousField = fields[fields.size() - 1];
                //текущая точка - предыдущая - вектор силы
                ElectroField field(magintPointSize,10,1,fieldPos - previousField.position,fieldPos);
                fields.push_back(field);
            }
            else
               fields.push_back(ElectroField(magintPointSize,10,1,{0,0,0},fieldPos));
            
        }
        return fields;
    }
    void TestTube::setTubePoints()
    {
        for(size_t it = 0; it < tubeVert.size(); it+=3)
        {
            tubePoints.push_back({tubeVert[it],tubeVert[it+1],tubeVert[it+2]});
        }
    }

    void TestTube::Init(std::string shaderPath)
    {
        addParticle({-80,20,0},1,1,particleDefaultVelocity);

        //строим верхнюю половину
        std::vector<float> firstSide = buildCircle(downSideTubeRadius,0,0,2,0.01);
        //строим нижнюю половину
        std::vector<float> secondSide = buildCircle(upperSideTubeRadius,2.5,2.5,1.8,0.01);

        //Строим точки по центру
        magnitPoints = buildCircle(middleTubeRad,1.25,1.25,1.8,0.3);
        //Строим магнитные поля на основе magnitPoints
        electroFields = getMagnitFields(magnitPoints,magintPointSize);

        //создаем единый массив для отрисовки всех точек разом
        tubeVert = firstSide;
        tubeVert.insert(tubeVert.end(),secondSide.begin(),secondSide.end());
        setTubePoints();
        //магия opengl
        AddVertexArray();
        AddBuffer(particles[0].getVertecies().begin()._Ptr,sizeof(float) * particles[0].getVertecies().size());

        AddVertexArray();
        AddBuffer(&tubeVert[0],sizeof(float) * tubeVert.size());

        AddVertexArray();
        AddBuffer(&magnitPoints[0],sizeof(float) * magnitPoints.size());
    }

    void TestTube::OnUpdate(//
        float deltaTime,
        float aspect,
        const glm::vec3& cameraPos,
        glm::mat4& pMat,
        glm::mat4& vMat)
    {
        Test::OnUpdate(deltaTime,aspect,cameraPos,pMat,vMat);

        particleSpawnTick(deltaTime);        
        drawParticles(deltaTime,vMat);
        drawTube(vMat);
    }
    void TestTube::addParticle(const glm::vec3& startPos,const float& radius,const float& charge,const glm::vec3& startVelocity)
    {
        glm::vec3 randPos{float( rand() % 15),float( rand() % 10), 0.f};
        particles.push_back(Particle(startPos + randPos,startVelocity,36,18,radius,charge));
    }

    void TestTube::drawParticles(float deltaTime,glm::mat4 vMat)
    {
        for(auto& particle : particles)
        {
            if(particle.getPosition().x < 80 && particle.getPosition().y < 50 && particle.getDidParticleMoveThroughField())
             {
                moveParticle(particle,deltaTime,vMat);
                glDrawParticle(particle);  

                if(DRAW_PARTICLE_PATH)
                    drawParticlePath(particle.getPosition());
             }//
        }

        if(DRAW_PARTICLE_PATH)
            for(auto& particle : pathParticles)
            {
                glm::mat4 translation = glm::translate(vMat,particle.getPosition());
                getShader().SetUnformMat4f("mv_matrix",translation);
                getShader().SetUniform4f("additionalColor",1,1,1,1);
                glDrawParticle(particle);
            }
    }
    void TestTube::glDrawParticle(const Particle& particle)
    {
        //вжух и оно рисует
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

    void TestTube::drawTube(glm::mat4 vMat)
    {
        GLCall(glPointSize(1));
        EnableVertexArray(1);

        getShader().SetUnformMat4f("mv_matrix",vMat);
        
        
        getShader().SetUniform4f("additionalColor",1,1,1,1);
        GLCall(glDrawArrays(GL_POINTS,0,tubeVert.size()));
        

        getShader().SetUniform4f("additionalColor",1,0,0,1);
        EnableVertexArray(2);
        GLCall(glPointSize(3));
        GLCall(glDrawArrays(GL_POINTS,0,magnitPoints.size()));
    }

    void TestTube::particleSpawnTick(float DeltaTime)
    {
        if(particleSpawnTimer <= 0)
        {
            particleSpawnTimer = particleSpawnTime;
            float charge = rand() % 100 > 100 ? -1 : 1;
            addParticle({-80,0,0},0,charge,particleDefaultVelocity);
        }
        else
            particleSpawnTimer -= DeltaTime;
    }

        void TestTube::moveParticle(Particle& particle,float deltaTime, glm::mat4 vMat)
        {
            for(auto& field : electroFields) // Check distance to all fields to detect a collision.
            {
                float pointsDist = glm::distance(particle.getPosition(),field.position);
                if(pointsDist < field.radius) 
                {

                    glm::vec3 inc = field.powerVector / particle.getWeight();
                    if(inc != glm::vec3())
                    {
                        particle.setVelocity(inc);
                        particle.isParticleAffectedByField = true;
                    }

                }
            }
            checkDidParticleHittedTube(particle);
            getShader().SetUnformMat4f("mv_matrix",calcTranslation(vMat,particle.getPosition()));
            glm::vec3 inc = multiplyVecByScalar(particle.getVeclocity(),particle.getSpeed());
            particle.movePosition(inc);
        }

        void TestTube::checkDidParticleHittedTube(Particle& outParticle)
        {
            for(auto& side : tubePoints)
            {
            
               if(glm::length(outParticle.getPosition() - side) < 1)
                    outParticle.setDidParticleMoveThroughField(false);
            }
        }

        void TestTube::OnTestEnd()
        {
            size_t aliveParticles = 0;
            size_t deadParticles = 0;
            for(auto& particle : particles)
            {
                if(particle.getDidParticleMoveThroughField() && particle.isParticleAffectedByField)
                    aliveParticles++;
                else if(particle.getDidParticleMoveThroughField())
                    deadParticles++;
            }
        //
            std::cout
            <<"Magnit field size: "
            <<magintPointSize
            <<"Alive particles "
            <<aliveParticles
            << "Dead particles: "
            << deadParticles
            <<" With time: "
            <<glfwGetTime()
            <<std::endl;
        }

}