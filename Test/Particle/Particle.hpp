#pragma once
#include "Math.hpp"
#include "Types.hpp"
#include "Vector.hpp"


namespace Test
{
class Particle
{
public:
	Particle(
	    TVec3 pos,
	    TVec3 vel,
	    uint32 _stackCount,
	    uint32 _sectorCount,
	    uint32 _radius,
	    float _charge);

	~Particle();

	inline void setColor(const TVec3 _color)
	{
		color = _color;
	}

	inline const TVec3& getColor() const
	{
		return color;
	}

	inline float getCharge() const
	{
		return charge;
	}

	inline void increaseSpeed(float Inc)
	{
		speed += Inc;
	}

	inline void IncreaseRotSpeed(float Inc)
	{
		rotationSpeed += Inc;
	}

	inline void move()
	{
		position += velocity * speed;
	}

	inline void movePosition(TVec3 vel)
	{
		position += vel;
	}

	inline void incVelocity(TVec3 inc)
	{
		velocity += inc;
	}

	inline void setVelocity(TVec3 inc)
	{
		velocity = inc;
	}

	inline TVec3 getVeclocity() const
	{
		return velocity;
	}

	inline float getSpeed() const
	{
		return speed;
	}

	inline TMat4 rotate(float deltaTime)
	{
		currentRotationAngle += deltaTime * rotationSpeed;
		if (currentRotationAngle > 360)
			currentRotationAngle = 0;
		return glm::rotate(TMat4(1.0f), float(currentRotationAngle), TVec3(1.0, 1.0, 1.0));
	}

	inline const TVec3& getPosition() const
	{
		return position;
	}

	inline uint32 getStackCount() const
	{
		return stackCount;
	}

	inline uint32 getSectorCount() const
	{
		return sectorCount;
	}

	inline uint32 getRadius() const
	{
		return radius;
	}

	inline OVector<float> getVertecies() const
	{
		return vertices;
	}

	inline void SetCharge(float Value)
	{
		charge = Value;
	}

	inline bool getDidParticleMoveThroughField() const
	{
		return DidParticleMoveThroughField;
	}

	inline void setDidParticleMoveThroughField(bool Value)
	{
		DidParticleMoveThroughField = Value;
	}

	void updateColor();
	bool isParticleAffectedByField = false;

private:
	void addVertex(float x, float y, float z);
	void addTexCoord(float s, float t);
	void addIndices(unsigned int i1, unsigned int i2, unsigned int i3);
	OVector<float> computeFaceNormal(float x1, float y1, float z1, // v1
	                                 float x2, float y2, float z2, // v2
	                                 float x3, float y3, float z3); // v3
	void clearArrays();
	void addNormal(float nx, float ny, float nz);
	void createVertecies();
	void buildInterleavedVertices();

	TVec3 color{ 1.f, 1.f, 1.f };
	TVec3 position{ 0.f, 0.f, 0.f };
	TVec3 velocity;

	uint32 stackCount = 12;
	uint32 sectorCount = 36;
	uint32 radius = 1;

	bool DidParticleMoveThroughField = true;
	float speed = 1.f;

	OVector<float> vertices;
	OVector<float> normals;
	OVector<float> texCoords;
	OVector<int32_t> indices;
	OVector<int32_t> lineIndices;

	float weight = 1.f;
	float currentRotationAngle = 0.f;
	float rotationSpeed = 10.f;
	float charge = -1.f;

	// interleaved
	OVector<float> interleavedVertices;
	int interleavedStride;
};

} // namespace Test
