#pragma once

#include "Test.hpp"

namespace Test
{
    class TestSimpleBox : public Test
    {

    public:
        TestSimpleBox() = default;
        TestSimpleBox(String shaderPath);

        void OnUpdate(
            float deltaTime,
            float aspect,
            const Vec3 &cameraPos,
            Mat4 &pMat,
            Mat4 &vMat) override;

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