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
	UpdateCameraDirection();
}

void OCamera::SetTarget(const OVec3& Arg)
{
	CameraTarget = Arg;
	UpdateCameraDirection();
}

void OCamera::Init()
{
	Rotate(0, 0);
	CameraPosition = { -5, 0, 0 };
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

	RAPI_LOG(Log, "Camera is rotated! Current Pitch:{} Yaw:{}", TO_STRING(Pitch), TO_STRING(Yaw));
}
void OCamera::Translate(ETranslateDirection Dir)
{
	OVec3 delta;
	switch (Dir)
	{
	case ETranslateDirection::Forward:
		delta = FrontVector * CameraSpeed;
		break;
	case ETranslateDirection::Backward:
		delta = -FrontVector * CameraSpeed;
		break;
	case ETranslateDirection::Left:
		delta = -glm::normalize(glm::cross(FrontVector, UpVector)) * CameraSpeed;
		break;
	case ETranslateDirection::Right:
		delta = glm::normalize(glm::cross(FrontVector, UpVector)) * CameraSpeed;
		break;
	case ETranslateDirection::Up:
		delta = UpVector * CameraSpeed;
		break;
	case ETranslateDirection::Down:
		delta = -UpVector * CameraSpeed;
		break;
	}
	CameraPosition += delta;
	UpdateCameraDirection();
}

void OCamera::Tick(float DeltaTime)
{
	CameraSpeed = CameraSpeedMultiplier * DeltaTime;
}
OMat4 OCamera::GetCameraView() const
{
	return glm::lookAt(CameraPosition, CameraPosition + FrontVector, CameraUp);
}

} // namespace RAPI
