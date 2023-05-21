//
// Created by Cybea on 3/6/2023.
//

#ifndef RENDERAPI_CAMERA_HPP
#define RENDERAPI_CAMERA_HPP
#include "Math.hpp"
#include "Types.hpp"
#include "Utils/Types/Threads/Thread.hpp"
namespace RAPI
{
enum class ETranslateDirection
{
	Forward,
	Backward,
	Left,
	Right,
	Up,
	Down
};

class OCamera
{
public:
	OCamera() = default;

	void SetPosition(const OVec3& Arg);
	NODISCARD FORCEINLINE const OVec3& GetPosition() const
	{
		return CameraPosition;
	}

	void SetTarget(const OVec3& Arg);
	void Translate(ETranslateDirection Dir);
	void Rotate(float XOffset, float YOffset);
	void Tick(float DeltaTime);
	OMat4 GetCameraView() const;

	void Init();

private:
	void UpdateCameraDirection();

	float Sensitivity = 1.F;
	float CameraSpeed = 1.F;
	float CameraSpeedMultiplier = 10.F;

	OVec3 CameraPosition;
	OVec3 CameraTarget;
	OVec3 CameraDirection;

	OVec3 CameraRight;
	OVec3 CameraUp;
	OVec3 CameraFront{ 0, 0, -1.f };

	OVec3 UpVector{ 0, 1.f, 0 };

	float Yaw{ 0 };
	float Pitch{ 0 };
};

} // namespace RAPI

#endif // RENDERAPI_CAMERA_HPP
