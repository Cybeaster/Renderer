#pragma once

#include "Test.hpp"

namespace Test
{
    class TestSimpleBox : public Test
    {

    public:
        TestSimpleBox() = default;
        TestSimpleBox(TString shaderPath);

        void OnUpdate(
            float deltaTime,
            float aspect,
            const TVec3 &cameraPos,
            TMat4 &pMat,
            TMat4 &vMat) override;

        float box[18]{
            0.f,
            1.f,
            1.f,
            1.f,
            1.f,
            1.f,
        };
    };

}