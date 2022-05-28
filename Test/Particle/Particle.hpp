#pragma once
#include <glm.hpp>
#include <gtc/matrix_transform.hpp>
#include <gtc/type_ptr.hpp>
#include <vector>


namespace test
{
    class Particle
    {
    public:

        Particle(
            glm::vec3 pos,
            glm::vec3 vel,
            uint32_t _stackCount,
            uint32_t _sectorCount,
            uint32_t _radius,
            float _charge);

        ~Particle();
        inline void setColor(const glm::vec3 _color)
        {color = _color;}

        inline const glm::vec3& getColor()const
        {return color;}
        
        inline float getCharge()const
        {return charge;}

        inline void increaseSpeed(float Inc)
        {speed += Inc;}
        
        inline void increaseRotSpeed(float Inc)
        {rotationSpeed += Inc;}
        
        inline void move()
        {
            position += velocity * speed;
        }

        inline void incVelocity(glm::vec3 inc)
        {velocity += inc;}

        inline glm::mat4 rotate(float deltaTime)
        {
           
            currentRotationAngle += deltaTime * rotationSpeed;
            if(currentRotationAngle > 360)
                currentRotationAngle = 0;
            return glm::rotate(glm::mat4(1.0f),float(currentRotationAngle),glm::vec3(1.0,1.0,1.0)); 
        }

        inline const glm::vec3& getPosition() const
        {return position;}

        inline uint32_t getStackCount()const
        {return stackCount;}
        
        inline uint32_t getSectorCount()const
        {return sectorCount;}

        inline uint32_t getRadius()const
        {return radius;}

        inline std::vector<float> getVertecies()const
        {return vertices;}
        
        inline void setCharge(float Value)
        {charge = Value;}

        void updateColor();
    private:
    
        void addVertex(float x, float y, float z);
        void addTexCoord(float s, float t);
        void addIndices(unsigned int i1, unsigned int i2, unsigned int i3);
        std::vector<float> computeFaceNormal( float x1, float y1, float z1,  // v1
                                                        float x2, float y2, float z2,  // v2
                                                        float x3, float y3, float z3);  // v3
        void clearArrays();
        void addNormal(float nx, float ny, float nz);
        void createVertecies();
        void buildInterleavedVertices();


        glm::vec3 color{1.f,1.f,1.f};
        glm::vec3 position{0.f,0.f,0.f};
        glm::vec3 velocity;
        
        uint32_t stackCount = 12;
        uint32_t sectorCount = 36;
        uint32_t radius = 1.f;
        
        float speed = 1.f;

        std::vector<float> vertices;
        std::vector<float> normals;
        std::vector<float> texCoords;
        std::vector<int32_t> indices;
        std::vector<int32_t> lineIndices;

        float currentRotationAngle = 0.f;
        float rotationSpeed = 10.f;
        float charge = -1.f;

        // interleaved
        std::vector<float> interleavedVertices;
        int interleavedStride;      
    };
    
    
    
} // namespace test
