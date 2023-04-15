//
// Created by Cybea on 3/6/2023.
//

#include "Camera.hpp"

namespace RAPI
{
void OCamera::SetPosition(const OVec3& Arg)
{
	OMutexGuard guard(TargetMutex);

	CameraPosition = Arg;
	UpdateCameraDirection();
}

void OCamera::SetTarget(const OVec3& Arg)
{
	OMutexGuard guard(TargetMutex);

	CameraTarget = Arg;
	UpdateCameraDirection();
}

void OCamera::UpdateCameraDirection()
{
	CameraDirection = glm::normalize(CameraPosition - CameraTarget);
	CameraRight = glm::normalize(glm::cross(UpVector, CameraDirection));
	CameraUp = glm::normalize(glm::cross(CameraDirection, CameraRight));
}
void OCamera::Rotate(float XOffset, float YOffset)
{
	OMutexGuard guard(RotateMutex);

	XOffset *= Sensitivity;
	YOffset *= Sensitivity;

	Yaw += XOffset;
	Pitch += YOffset;

	if (Pitch > 89)
		Pitch = 89;

	if (Pitch < -89)
		Pitch = -89;

	auto yawRadians = SMath::ToRadians(Yaw);
	auto pitchRadians = SMath::ToRadians(Pitch);

	OVec3 direction;
	direction.x = cos(yawRadians) * cos(pitchRadians);
	direction.y = sin(pitchRadians);
	direction.z = sin(yawRadians) * cos(pitchRadians);

	FrontVector = glm::normalize(direction);
}
void OCamera::Translate(ETranslateDirection Dir)
{
	OVec3 delta;
	switch (Dir)
	{
	case Forward:
		delta = FrontVector * CameraSpeed;
		break;
	case Backward:
		delta = -FrontVector * CameraSpeed;
		break;
	case Left:
		delta = -glm::normalize(glm::cross(FrontVector, UpVector)) * CameraSpeed;
		break;
	case Right:
		delta = glm::normalize(glm::cross(FrontVector, UpVector)) * CameraSpeed;
		break;
	case Up:
		delta = UpVector * CameraSpeed;
		break;
	case Down:
		delta = -UpVector * CameraSpeed;
		break;
	}
	OMutexGuard guard(TargetMutex);
	CameraPosition += delta;
}

void OCamera::Tick(float DeltaTime)
{
	CameraSpeed = CameraSpeedMultiplier * DeltaTime;
}

} // namespace RAPI
