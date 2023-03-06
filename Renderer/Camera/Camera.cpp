//
// Created by Cybea on 3/6/2023.
//

#include "Camera.hpp"

namespace RenderAPI
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
}

void OCamera::Translate(const OVec3& Delta)
{
	CameraPosition += FrontVector * Delta;
}

} // namespace RenderAPI
