#pragma once

#include "Types.hpp"
namespace RAPI
{

class OWindow
{
public:
	virtual ~OWindow() = default;
	virtual void InitWindow() = 0;
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

protected:
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
