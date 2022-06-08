#pragma once
#include <vector>
#include "Test.hpp"
#include "Particle.hpp"
#include "ElectroField.hpp"
#include <vector>
namespace test
{

    
    class TestTube : public Test
    {

    private:
        void OnTestEnd() override;
        void drawTube(glm::mat4 vMat);
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
        const glm::vec3 particleDefaultVelocity{1.f,0,0};
        const float defaultFieldStrenght = 1.1f;

        
        //настройка трубы
        //радиус (положение нижней части)
        float downSideTubeRadius = 1;
        //радиус (положение верхней части)
        float upperSideTubeRadius = 5;
        float middleTubeRad = (downSideTubeRadius + upperSideTubeRadius) / 2;
        float magintPointSize = upperSideTubeRadius - downSideTubeRadius;

        void checkDidParticleHittedTube(Particle& outParticle);
        void setTubePoints();
    public:

        std::vector<ElectroField> electroFields;
        std::vector<float> tubeVert;
        std::vector<glm::vec3> tubePoints;
        std::vector<float> magnitPoints;
        
        TestTube(std::string shaderPath);
        void Init(std::string shaderPath);
        TestTube() = default;

        void OnUpdate(
        float deltaTime,
        float aspect,
        const glm::vec3& cameraPos,
        glm::mat4& pMat,
        glm::mat4& vMat) override;
        
    };
    
} // namespace test
