#pragma once

#include "Types.hpp"
namespace RAPI
{

class OWindow
{
	virtual double GetDeltaTime() const = 0;

	virtual ~OWindow() = default;
protected:
	virtual bool NeedClose() = 0;
	virtual void InitWindow() = 0;
	virtual void DrawStart() = 0;
	virtual void DrawEnd() = 0;
	virtual void CalcDeltaTime() = 0;
	virtual void CalcAspect() = 0;

	int32 Width{ 0 };
	int32 Height{ 0 };

	double PreviousTime{ 0 };
	double DeltaTime{ 0 };

	double AspectRatio{ 0 };

	const OString WindowName = "RAPI";
};

} // namespace RAPI
