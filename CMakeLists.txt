cmake_minimum_required(VERSION 3.21)

project(RenderAPI LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED)
set(Boost_DEBUG 1)

if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    add_compile_options(/EHsc)
endif ()
set(Boost_USE_STATIC_LIBS ON)

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
        ${PROJECT_SOURCE_DIR}/RendererTest/Particle
        ${PROJECT_SOURCE_DIR}/RendererTest
        ${PROJECT_SOURCE_DIR}/RendererTest/Tube
        ${PROJECT_SOURCE_DIR}/RendererTest/Tube/Plane
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
find_package(Boost COMPONENTS thread REQUIRED)
add_executable(exec
        main.cpp
        Renderer/Renderer.cpp
        Renderer/Vertex/Buffer.cpp
        Renderer/Vertex/VertexAttribBuffer.cpp
        InputHandlers/InputHandler.cpp
        InputHandlers/RendererInputHandler.cpp
        Application/Application.cpp
        Shader/Shader.cpp
        Texture/Texture.cpp
        RendererTest/Test.cpp
        RendererTest/TestTexture.cpp
        RendererTest/TestSimpleCube.cpp
        RendererTest/Particle/Particle.cpp
        RendererTest/Particle/TestParticles.cpp
        RendererTest/TestSimpleSolarSystem.cpp
        Renderer/Vertex/VertexArrayElements/VertexArrayElem.cpp
        Renderer/Vertex/VertexArray.cpp


        Utils/Delegate/MulticastDelegate.hpp
        Utils/Types/Vector.hpp
        Utils/Types/Queue/ThreadSafeQueue.hpp
        Utils/Threads/ThreadPool/CustomTestThreadPool/ThreadPool.cpp
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
        Renderer/Camera/Camera.cpp Renderer/Camera/Camera.hpp
        Tests/ThreadTests.cpp
        Tests/ThreadTests.hpp
        Tests/Clocks.hpp
        Utils/Alogs/QuickSort.hpp
        Utils/Types/List.hpp
        Utils/Threads/Async.hpp
        Utils/UnitTests/TestGroup.cpp
        Utils/UnitTests/TestGroup.hpp
        Utils/Types/Stack/LockBasedStack.hpp Utils/Types/Queue/Queue.hpp Utils/Types/Queue.hpp Utils/Types/HashMap/ThreadSafeHashMap.hpp Utils/Types/List/ThreadSafeList.hpp Utils/Types/Stack/LockFreeStack.hpp Utils/Types/Stack/Stack.hpp Utils/Threads/HazardPointer.hpp Utils/Types/Stack/SharedPtrLockFreeStack.hpp Utils/Types/Stack/ReferenceBasedLockFreeStack.hpp Utils/Types/Queue/LockFreeQueue.hpp Utils/Types/Queue/LockFreeQueue.hpp Utils/Alogs/Sorter.hpp Utils/Types/Time.hpp Utils/Alogs/Accumulate.hpp Utils/Alogs/Accumulate.hpp Utils/Alogs/ForEach.hpp Utils/Threads/Utils.hpp Utils/Alogs/Find.hpp Utils/Alogs/PartialSum.hpp Utils/Alogs/Utils.hpp Utils/Alogs/Utils.cpp Utils/Threads/Utils.cpp Utils/Profiler/Profiler.hpp Utils/Profiler/StatGroup.hpp Utils/Threads/Barrier.hpp Utils/Threads/ThreadPool/SimpleThreadPool/ThreadPool.cpp Utils/Threads/ThreadPool/SimpleThreadPool/ThreadPool.hpp Utils/Threads/ThreadPool/ThreadPool.hpp Utils/Types/Queue/WorkStealingQueue.hpp Utils/Threads/InterruptibleThread/InterruptibleThread.cpp Utils/Threads/InterruptibleThread/InterruptibleThread.hpp Utils/Threads/InterruptibleThread/InterruptFlag.hpp Utils/Threads/InterruptibleThread/InterruptFlag.cpp Utils/Threads/JoinThreads.hpp Utils/Threads/SpinlockMutex.hpp Utils/Profiler/StatObject.hpp Utils/Types/ShaderName.hpp Utils/Profiler/Profiler.cpp Utils/Profiler/StatGroup.cpp InputHandlers/ConsoleInputHandler.cpp InputHandlers/ConsoleInputHandler.hpp Window/GlfwWindow.cpp Window/GlfwWindow.hpp Window/Window.hpp Utils/Threads/ThreadPool/NamedThreadPool/NamedThreadPool.cpp Utils/Threads/ThreadPool/NamedThreadPool/NamedThreadPool.hpp ModelsImporter/OBJImporter/ObjImporter.cpp ModelsImporter/OBJImporter/ObjImporter.hpp Models/GeneratedModel.hpp Models/GeneratedMode.cpp ModelsImporter/ModelImporter.hpp RendererTest/TestModelling.cpp RendererTest/TestModelling.hpp Models/Model.cpp ModelsImporter/ModelImporter.cpp ModelsImporter/ImportManager.cpp ModelsImporter/ImportManager.hpp Utils/Types/Keys/KeyState.hpp)


set(GLFW_LIB ${PROJECT_SOURCE_DIR}/Externals/Lib/glfw3.lib)
set(GLEW_LIB ${PROJECT_SOURCE_DIR}/Externals/Lib/glew32.lib)
set(SOIL_LIB ${PROJECT_SOURCE_DIR}/Externals/Lib/soil2-debug.lib)

set(GLFW_DLL ${PROJECT_SOURCE_DIR}/Externals/dll/glfw3.dll)
set(GLEW_DLL ${PROJECT_SOURCE_DIR}/Externals/dll/glew64.dll)
set(GLEW32_DLL ${PROJECT_SOURCE_DIR}/Externals/dll/glew32.dll)


link_directories(${Boost_LIBRARY_DIR})
target_link_libraries(
        exec
        PRIVATE
        ${GLFW_LIB}
        ${OPENGL_LIBRARY}
        ${GLEW_LIB}
        ${SOIL_LIB}
        ${Boost_LIBRARIES}
)
