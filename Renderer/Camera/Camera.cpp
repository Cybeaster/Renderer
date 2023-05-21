//
// Created by Cybea on 3/6/2023.
//

#include "Camera.hpp"

#include "Logging/Log.hpp"

namespace RAPI
{
void OCamera::SetPosition(const OVec3& Arg)
{
	CameraPosition = Arg;
}

void OCamera::SetTarget(const OVec3& Arg)
{
	CameraTarget = Arg;
	UpdateCameraDirection();
}

void OCamera::Init()
{
	CameraTarget = { 0, 0, -1 };
	UpdateCameraDirection();
	CameraPosition = { 0, 3, 0 };
}

void OCamera::UpdateCameraDirection()
{
	CameraDirection = glm::normalize(CameraPosition - CameraTarget);
	CameraRight = glm::normalize(glm::cross(UpVector, CameraDirection));
	CameraUp = glm::normalize(glm::cross(CameraDirection, CameraRight));
}
void OCamera::Rotate(float XOffset, float YOffset)
{
	XOffset *= Sensitivity;
	YOffset *= Sensitivity;

	Yaw = fmod(Yaw + XOffset, 360);

	Pitch += YOffset;

	if (Pitch > 180)
		Pitch = 180;

	if (Pitch < -180)
		Pitch = -180;

	auto yawRadians = SMath::ToRadians(Yaw);
	auto pitchRadians = SMath::ToRadians(Pitch);

	OVec3 direction;
	direction.x = cos(yawRadians) * cos(pitchRadians);
	direction.y = sin(pitchRadians);
	direction.z = sin(yawRadians) * cos(pitchRadians);

	CameraFront = glm::normalize(direction);
}

void OCamera::Translate(ETranslateDirection Dir)
{
	OVec3 delta;
	switch (Dir)
	{
	case ETranslateDirection::Forward:
		delta = CameraFront * CameraSpeed;
		break;
	case ETranslateDirection::Backward:
		delta = -CameraFront * CameraSpeed;
		break;
	case ETranslateDirection::Left:
		delta = -glm::normalize(glm::cross(CameraFront, UpVector)) * CameraSpeed;
		break;
	case ETranslateDirection::Right:
		delta = glm::normalize(glm::cross(CameraFront, UpVector)) * CameraSpeed;
		break;
	case ETranslateDirection::Up:
		delta = UpVector * CameraSpeed;
		break;
	case ETranslateDirection::Down:
		delta = -UpVector * CameraSpeed;
		break;
	}
	CameraPosition += delta;
}

void OCamera::Tick(float DeltaTime)
{
	CameraSpeed = CameraSpeedMultiplier * DeltaTime;
}
OMat4 OCamera::GetCameraView() const
{
	return glm::lookAt(CameraPosition, CameraPosition + CameraFront, CameraUp);
}

} // namespace RAPI
