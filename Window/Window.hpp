#pragma once

#include "Math.hpp"
#include "Types.hpp"
namespace RAPI
{

class OWindow
{
public:
	virtual ~OWindow() = default;
	virtual void InitWindow();
	virtual bool NeedClose() = 0;
	virtual void DrawStart() = 0;
	virtual void DrawEnd() = 0;

	FORCEINLINE auto GetDeltaTime() const
	{
		return DeltaTime;
	}

	FORCEINLINE auto GetAspectRation() const
	{
		return AspectRatio;
	}

	OVec2 GetNDC() const;

	void OnMousePosition(double NewX, double NewY);

protected:
	virtual void CalcDeltaTime() = 0;
	virtual void CalcAspect() = 0;

	float Width{ 0 };
	float Height{ 0 };

	double PreviousTime{ 0 };
	double DeltaTime{ 0 };

	double AspectRatio{ 0 };

	OVec2 MousePosition{ 0, 0 };

	const OString WindowName = "RAPI";
};

} // namespace RAPI
