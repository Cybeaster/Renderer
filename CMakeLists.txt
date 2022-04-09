cmake_minimum_required(VERSION 3.21)

project(Opengl)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED)

find_package(OpenGL REQUIRED)

include_directories(
	${PROJECT_SOURCE_DIR}/Externals/glfw/include
	${PROJECT_SOURCE_DIR}/Externals/glm
	${PROJECT_SOURCE_DIR}/Externals/glew/include
	${PROJECT_SOURCE_DIR}/Renderer
	${PROJECT_SOURCE_DIR}/VertexArray
	${PROJECT_SOURCE_DIR}/Shader
	${PROJECT_SOURCE_DIR}/Texture
	${PROJECT_SOURCE_DIR}/Externals/Vendor
	${PROJECT_SOURCE_DIR}/Externals/imgui
	${PROJECT_SOURCE_DIR}/Test
		)


set(GLFW_LIB ${PROJECT_SOURCE_DIR}/Externals/glfw/lib-vc2019/glfw3.lib)
set(GLEW_LIB ${PROJECT_SOURCE_DIR}/Externals/glew/lib/Release/x64/glew32s.lib)


add_executable(exec 
main.cpp
Application/Application.cpp
Renderer/Renderer.cpp
Renderer/VertexBuffer.cpp
Renderer/IndexBuffer.cpp
VertexArray/VertexArray.cpp
Renderer/VertexBufferLayout.cpp
Shader/Shader.cpp
Texture/Texture.cpp
Externals/imgui/imgui.cpp
Externals/imgui/imgui_impl_glfw_gl3.cpp
Externals/imgui/imgui_demo.cpp
Externals/imgui/imgui_draw.cpp
Test/TestClearColor.cpp
Test/Test.cpp
Test/TestTexture.cpp
Test/TestSimpleCube.cpp
)

target_link_libraries(exec ${GLFW_LIB} ${OPENGL_LIBRARY} ${GLEW_LIB})