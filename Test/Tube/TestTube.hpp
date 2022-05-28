#pragma once
#include <vector>
#include "Test.hpp"
#include "Particle.hpp"
#include "ElectroField.hpp"
#include "Tube.hpp"
namespace test
{
    
    class TestTube : public Test
    {

    private:
        void drawTube();
        void glDrawParticle(const Particle& particle);
        void drawParticlePath(const glm::vec3& pos);
        void addParticle(const glm::vec3& startPos,const float& radius,const float& charge,const glm::vec3& startVelocity);
         /**
         * @brief Считает тамер для спавна частиц.
         * 
         * @param DeltaTime 
         */
        void particleSpawnTick(float DeltaTime);

        /**
         * @brief Отрисовывает частицы.
         * 
         * @param deltaTime Время между кадрами.
         * @param vMat Матрица камеры (описывает положение оной в пространстве).
         */
        void drawParticles(float deltaTime,glm::mat4 vMat);


        /**
         * @brief Сдвигает частицу на определенную дистацию каждый кадр.
         * 
         * @param particle Частица, которую необходимо сдвинуть.
         * @param deltaTime Время между кадрами.
         * @param vMat Матрица камеры (описывает положение оной в пространстве).
         */
        void moveParticle(Particle& particle,float deltaTime, glm::mat4 vMat);

        /**
         * @brief Заспавненные частицы.
         * 
         */
        std::vector<Particle> particles;

        /**
         * @brief Заспавненные частицы отображающие путь.
         * 
         */
        std::vector<Particle> pathParticles;

        /**
         * @brief Таймер для отсчета спавна частиц.
         * 
         */
        float particleSpawnTimer = 0.f;
        /**
         * @brief Время, через которое будут спавнится частицы.
         * 
         */
        float particleSpawnTime = 0.5f;
       

        /**
         * @brief Начальная скорость всех частиц по x,y и z соответственно.
         * 
         */
        const glm::vec3 particles45StartVel{0.95f,0.1f,0};
        const glm::vec3 particlesNegative45StartVel{0.95f,-0.5f,0};

        const float defaultFieldStrenght = 1.1f;

        Tube magnitTube;
    public:

        TestTube(std::string shaderPath);
        ~TestTube();

        void OnUpdate(GLFWwindow* window,
        float deltaTime,
        float aspect,
        const glm::vec3& cameraPos,
        glm::mat4& pMat,
        glm::mat4& vMat) override;
        
    };
    
} // namespace test
