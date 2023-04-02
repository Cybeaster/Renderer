cmake_minimum_required(VERSION 3.21)

project(RenderAPI LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED)
set(Boost_DEBUG 1)

include_directories(
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/Externals/Include/glfw
        ${PROJECT_SOURCE_DIR}/Externals/Include/glm
        ${PROJECT_SOURCE_DIR}/Externals/Include/glew
        ${PROJECT_SOURCE_DIR}/Externals/Include/SOIL2
        ${PROJECT_SOURCE_DIR}/Externals/Include/cuda
        ${PROJECT_SOURCE_DIR}/Renderer
        ${PROJECT_SOURCE_DIR}/Renderer/Removed
        ${PROJECT_SOURCE_DIR}/Shader
        ${PROJECT_SOURCE_DIR}/Texture
        ${PROJECT_SOURCE_DIR}/Externals/Vendor
        ${PROJECT_SOURCE_DIR}/Test/Particle
        ${PROJECT_SOURCE_DIR}/Test
        ${PROJECT_SOURCE_DIR}/Test/Tube
        ${PROJECT_SOURCE_DIR}/Test/Tube/Plane
        ${PROJECT_SOURCE_DIR}/Utils
        ${PROJECT_SOURCE_DIR}/Externals/freeglut/include
        ${PROJECT_SOURCE_DIR}/Utils/Threads/ThreadPool
        ${PROJECT_SOURCE_DIR}/Threads
        ${PROJECT_SOURCE_DIR}/Utils/Threads/HierarchicalMutex
        ${PROJECT_SOURCE_DIR}/Utils/Delegate
        ${PROJECT_SOURCE_DIR}/Utils/Checks
        ${PROJECT_SOURCE_DIR}/Utils/Types
        ${PROJECT_SOURCE_DIR}/Utils/Types/Tuple
        ${PROJECT_SOURCE_DIR}/Utils/Types/Queue
        ${PROJECT_BINARY_DIR}/Utils/Allocators
        ${PROJECT_BINARY_DIR}/Utils/Delegate
        ${Boost_INCLUDE_DIR}
        }
)

find_package(OpenGL REQUIRED)
find_package(Boost REQUIRED)
add_executable(exec
        main.cpp
        Renderer/Renderer.cpp
        Renderer/Vertex/Buffer.cpp
        Renderer/Vertex/VertexAttribBuffer.cpp
        Renderer/InputHandlers/InputHandler.cpp
        Renderer/InputHandlers/RendererInputHandler.cpp
        Application/Application.cpp
        Shader/Shader.cpp
        Texture/Texture.cpp
        Test/Test.cpp
        Test/TestTexture.cpp
        Test/TestSimpleCube.cpp
        Test/Particle/Particle.cpp
        Test/Particle/TestParticles.cpp
        Test/TestSimpleSolarSystem.cpp
        Renderer/Vertex/VertexArrayElements/VertexArrayElem.cpp
        Renderer/Vertex/VertexArray.cpp


        Utils/Delegate/MulticastDelegate.hpp
        Utils/Types/Vector.hpp
        Utils/Types/Queue/ThreadSafeQueue.hpp
        Utils/Threads/ThreadPool/ThreadPool.cpp
        Utils/Threads/HierarchicalMutex/HierarchicalMutex.cpp
        Utils/Checks/Exception.hpp
        Utils/Logging/Log.hpp
        Models/Sphere.cpp
        Models/Sphere.hpp
        Models/Sphere.hpp
        Utils/Logging/Printer.hpp
        Models/Torus.cpp
        Models/Torus.hpp
        Models/Model.hpp
        Models/Model.cpp
        Renderer/Camera/Camera.cpp Renderer/Camera/Camera.hpp
        Utils/Threads/Tests/ThreadTests.cpp
        Utils/Threads/Tests/ThreadTests.hpp
        Utils/Threads/Tests/Clocks.hpp
        Utils/Alogs/QuickSort.hpp
        Utils/Types/List.hpp
        Utils/Threads/Async.hpp
        Utils/UnitTests/TestGroup.cpp
        Utils/UnitTests/TestGroup.hpp
        Utils/Types/Stack/LockBasedStack.hpp Utils/Types/Queue/Queue.hpp Utils/Types/Queue.hpp Utils/Types/HashMap/ThreadSafeHashMap.hpp Utils/Types/List/ThreadSafeList.hpp Utils/Types/Stack/LockFreeStack.hpp Utils/Types/Stack/Stack.hpp Utils/Threads/HazardPointer.hpp Utils/Types/Stack/SharedPtrLockFreeStack.hpp Utils/Types/Stack/ReferenceBasedLockFreeStack.hpp Utils/Types/Queue/LockFreeQueue.hpp Utils/Types/Queue/LockFreeQueue.hpp Utils/Alogs/Sorter.hpp Utils/Types/Time.hpp Utils/Alogs/Accumulate.hpp Utils/Alogs/Accumulate.hpp Utils/Alogs/ForEach.hpp Utils/Types/Threads/Utils.hpp Utils/Alogs/Find.hpp Utils/Alogs/PartialSum.hpp Utils/Alogs/Utils.hpp Utils/Alogs/Utils.cpp Utils/Types/Threads/Utils.cpp Utils/Profiler/Profiler.hpp Utils/Profiler/StatGroup.hpp Utils/Threads/Barrier.hpp)


set(GLFW_LIB ${PROJECT_SOURCE_DIR}/Externals/Lib/glfw3.lib)
set(GLEW_LIB ${PROJECT_SOURCE_DIR}/Externals/Lib/glew32.lib)
set(SOIL_LIB ${PROJECT_SOURCE_DIR}/Externals/Lib/soil2-debug.lib)

set(GLFW_DLL ${PROJECT_SOURCE_DIR}/Externals/dll/glfw3.dll)
set(GLEW_DLL ${PROJECT_SOURCE_DIR}/Externals/dll/glew64.dll)
set(GLEW32_DLL ${PROJECT_SOURCE_DIR}/Externals/dll/glew32.dll)


target_link_libraries(
        exec
        PRIVATE
        ${GLFW_LIB}
        ${OPENGL_LIBRARY}
        ${GLEW_LIB}
        ${SOIL_LIB}
        ${Boost_LIBRARIES}

)
