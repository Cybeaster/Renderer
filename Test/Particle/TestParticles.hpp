#pragma once
#include <vector>
#include "Test.hpp"
#include "Particle.hpp"
#include "ElectroField.hpp"
namespace test
{

    /**
     * @brief Спавнит частицы Particle в определенной зоне, с начальным ускоронием velocity, и поля, такие же объекты, с дополнительными параметрами вроде силы притяжения.
     * 
     * 
     */
    class TestParticles : public Test
    {

    private:

        void addField(const glm::vec3& pos,const float& strenght,const glm::vec3& chargeVec,const float& charge);

        void addParticle(const glm::vec3& startPos,const float& radius,const float& charge,const glm::vec3& startVelocity);
         /**
         * @brief Считает тамер для спавна частиц.
         * 
         * @param DeltaTime 
         */
        void particleSpawnTick(float DeltaTime);

        /**
         * @brief Считает тамер для спавна полей.
         * 
         * @param DeltaTime 
         */
        void fieldSpawnTick(float DeltaTime);

        /**
         * @brief Отрисовывает частицы.
         * 
         * @param deltaTime Время между кадрами.
         * @param vMat Матрица камеры (описывает положение оной в пространстве).
         */
        void drawParticles(float deltaTime,glm::mat4 vMat);

        /**
         * @brief Отрисовывает поля.
         * 
         * @param deltaTime Время между кадрами.
         * @param vMat Матрица камеры (описывает положение оной в пространстве).
         */
        void drawFields(float deltaTime,glm::mat4 vMat);

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
         * @brief Заспавненные поля.
         * 
         */
        std::vector<GravityField> electroFields;

        /**
         * @brief Таймер для отсчета спавна частиц.
         * 
         */
        float particleSpawnTimer = 0.f;
        /**
         * @brief Время, через которое будут спавнится частицы.
         * 
         */
        float particleSpawnTime = 0.05f;

        /**
         * @brief Таймер спавна полей.
         * 
         */
        float fieldSpawnTimer = 0.f;
        /**
         * @brief Время спавна полей.
         * 
         */
        float fieldSpawnTime = 100.f;
       

        /**
         * @brief Начальная скорость всех частиц по x,y и z соответственно.
         * 
         */
        const glm::vec3 particles45StartVel{0.95f,0.1f,0};
        const glm::vec3 particlesNegative45StartVel{0.95f,-0.5f,0};

        const float defaultFieldStrenght = 1.1f;
    public:
    
        TestParticles(std::string shaderPath);
        ~TestParticles();

        void OnUpdate(
        float deltaTime,
        float aspect,
        const glm::vec3& cameraPos,
        glm::mat4& pMat,
        glm::mat4& vMat) override;
        
    };
    
} // namespace test
