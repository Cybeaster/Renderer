#include "Particle.hpp"

namespace test
{

    Particle::Particle(
        glm::vec3 pos,
        glm::vec3 vel,
        uint32_t _stackCount,
        uint32_t _sectorCount,
        uint32_t _radius,
        float _charge)
        : position(pos),
          velocity(vel),
          stackCount(_stackCount),
          sectorCount(_sectorCount),
          radius(_radius),
          charge(_charge)
    {
        createVertecies();
    }



    Particle::~Particle()
    {
    }

    void Particle::createVertecies()
    {
        const float PI = acos(-1);
        // tmp vertex definition (x,y,z,s,t)
        struct Vertex
        {
            float x, y, z, s, t;
        };
        std::vector<Vertex> tmpVertices;

        float sectorStep = 2 * PI / sectorCount;
        float stackStep = PI / stackCount;
        float sectorAngle, stackAngle;

        // compute all vertices first, each vertex contains (x,y,z,s,t) except normal
        for (int i = 0; i <= stackCount; ++i)
        {
            stackAngle = PI / 2 - i * stackStep;  // starting from pi/2 to -pi/2
            float xy = radius * cosf(stackAngle); // r * cos(u)
            float z = radius * sinf(stackAngle);  // r * sin(u)

            // add (sectorCount+1) vertices per stack
            // the first and last vertices have same position and normal, but different tex coords
            for (int j = 0; j <= sectorCount; ++j)
            {
                sectorAngle = j * sectorStep; // starting from 0 to 2pi

                Vertex vertex;
                vertex.x = xy * cosf(sectorAngle); // x = r * cos(u) * cos(v)
                vertex.y = xy * sinf(sectorAngle); // y = r * cos(u) * sin(v)
                vertex.z = z;                      // z = r * sin(u)
                vertex.s = (float)j / sectorCount; // s
                vertex.t = (float)i / stackCount;  // t
                tmpVertices.push_back(vertex);
            }
        }

        // clear memory of prev arrays
        clearArrays();

        Vertex v1, v2, v3, v4; // 4 vertex positions and tex coords
        std::vector<float> n;  // 1 face normal

        int i, j, k, vi1, vi2;
        int index = 0; // index for vertex
        for (i = 0; i < stackCount; ++i)
        {
            vi1 = i * (sectorCount + 1); // index of tmpVertices
            vi2 = (i + 1) * (sectorCount + 1);

            for (j = 0; j < sectorCount; ++j, ++vi1, ++vi2)
            {
                // get 4 vertices per sector
                //  v1--v3
                //  |    |
                //  v2--v4
                v1 = tmpVertices[vi1];
                v2 = tmpVertices[vi2];
                v3 = tmpVertices[vi1 + 1];
                v4 = tmpVertices[vi2 + 1];

                // if 1st stack and last stack, store only 1 triangle per sector
                // otherwise, store 2 triangles (quad) per sector
                if (i == 0) // a triangle for first stack ==========================
                {
                    // put a triangle
                    addVertex(v1.x, v1.y, v1.z);
                    addVertex(v2.x, v2.y, v2.z);
                    addVertex(v4.x, v4.y, v4.z);

                    // put tex coords of triangle
                    addTexCoord(v1.s, v1.t);
                    addTexCoord(v2.s, v2.t);
                    addTexCoord(v4.s, v4.t);

                    // put normal
                    n = computeFaceNormal(v1.x, v1.y, v1.z, v2.x, v2.y, v2.z, v4.x, v4.y, v4.z);
                    for (k = 0; k < 3; ++k) // same normals for 3 vertices
                    {
                        addNormal(n[0], n[1], n[2]);
                    }

                    // put indices of 1 triangle
                    addIndices(index, index + 1, index + 2);

                    // indices for line (first stack requires only vertical line)
                    lineIndices.push_back(index);
                    lineIndices.push_back(index + 1);

                    index += 3; // for next
                }
                else if (i == (stackCount - 1)) // a triangle for last stack =========
                {
                    // put a triangle
                    addVertex(v1.x, v1.y, v1.z);
                    addVertex(v2.x, v2.y, v2.z);
                    addVertex(v3.x, v3.y, v3.z);

                    // put tex coords of triangle
                    addTexCoord(v1.s, v1.t);
                    addTexCoord(v2.s, v2.t);
                    addTexCoord(v3.s, v3.t);

                    // put normal
                    n = computeFaceNormal(v1.x, v1.y, v1.z, v2.x, v2.y, v2.z, v3.x, v3.y, v3.z);
                    for (k = 0; k < 3; ++k) // same normals for 3 vertices
                    {
                        addNormal(n[0], n[1], n[2]);
                    }

                    // put indices of 1 triangle
                    addIndices(index, index + 1, index + 2);

                    // indices for lines (last stack requires both vert/hori lines)
                    lineIndices.push_back(index);
                    lineIndices.push_back(index + 1);
                    lineIndices.push_back(index);
                    lineIndices.push_back(index + 2);

                    index += 3; // for next
                }
                else // 2 triangles for others ====================================
                {
                    // put quad vertices: v1-v2-v3-v4
                    addVertex(v1.x, v1.y, v1.z);
                    addVertex(v2.x, v2.y, v2.z);
                    addVertex(v3.x, v3.y, v3.z);
                    addVertex(v4.x, v4.y, v4.z);

                    // put tex coords of quad
                    addTexCoord(v1.s, v1.t);
                    addTexCoord(v2.s, v2.t);
                    addTexCoord(v3.s, v3.t);
                    addTexCoord(v4.s, v4.t);

                    // put normal
                    n = computeFaceNormal(v1.x, v1.y, v1.z, v2.x, v2.y, v2.z, v3.x, v3.y, v3.z);
                    for (k = 0; k < 4; ++k) // same normals for 4 vertices
                    {
                        addNormal(n[0], n[1], n[2]);
                    }

                    // put indices of quad (2 triangles)
                    addIndices(index, index + 1, index + 2);
                    addIndices(index + 2, index + 1, index + 3);

                    // indices for lines
                    lineIndices.push_back(index);
                    lineIndices.push_back(index + 1);
                    lineIndices.push_back(index);
                    lineIndices.push_back(index + 2);

                    index += 4; // for next
                }
            }
        }

        // generate interleaved vertex array as well
        buildInterleavedVertices();
    }

    void Particle::addVertex(float x, float y, float z)
    {
        vertices.push_back(x);
        vertices.push_back(y);
        vertices.push_back(z);
    }

    void Particle::addTexCoord(float s, float t)
    {
        texCoords.push_back(s);
        texCoords.push_back(t);
    }
    void Particle::addIndices(unsigned int i1, unsigned int i2, unsigned int i3)
    {
        indices.push_back(i1);
        indices.push_back(i2);
        indices.push_back(i3);
    }

    std::vector<float> Particle::computeFaceNormal(float x1, float y1, float z1,  // v1
                                                   float x2, float y2, float z2,  // v2
                                                   float x3, float y3, float z3)  // v3
    {
        const float EPSILON = 0.000001f;

        std::vector<float> normal(3, 0.0f);     // default return value (0,0,0)
        float nx, ny, nz;

        // find 2 edge vectors: v1-v2, v1-v3
        float ex1 = x2 - x1;
        float ey1 = y2 - y1;
        float ez1 = z2 - z1;
        float ex2 = x3 - x1;
        float ey2 = y3 - y1;
        float ez2 = z3 - z1;

        // cross product: e1 x e2
        nx = ey1 * ez2 - ez1 * ey2;
        ny = ez1 * ex2 - ex1 * ez2;
        nz = ex1 * ey2 - ey1 * ex2;

        // normalize only if the length is > 0
        float length = sqrtf(nx * nx + ny * ny + nz * nz);
        if(length > EPSILON)
        {
            // normalize
            float lengthInv = 1.0f / length;
            normal[0] = nx * lengthInv;
            normal[1] = ny * lengthInv;
            normal[2] = nz * lengthInv;
        }
        return normal;
    }
    void Particle::clearArrays()
    {
        vertices.clear();
        normals.clear();
        texCoords.clear();
        indices.clear();
        lineIndices.clear();

    }

    void Particle::addNormal(float nx, float ny, float nz)
    {
        normals.push_back(nx);
        normals.push_back(ny);
        normals.push_back(nz);
    }

    void Particle::buildInterleavedVertices()
    {
        std::vector<float>().swap(interleavedVertices);

        std::size_t i, j;
        std::size_t count = vertices.size();
        for(i = 0, j = 0; i < count; i += 3, j += 2)
        {
            interleavedVertices.push_back(vertices[i]);
            interleavedVertices.push_back(vertices[i+1]);
            interleavedVertices.push_back(vertices[i+2]);

            interleavedVertices.push_back(normals[i]);
            interleavedVertices.push_back(normals[i+1]);
            interleavedVertices.push_back(normals[i+2]);

            interleavedVertices.push_back(texCoords[j]);
            interleavedVertices.push_back(texCoords[j+1]);
        }
    }

    void Particle::updateColor()
    {
        float red = velocity.x;
        float blue = velocity.y;
        float green = velocity.z;
        
        red = glm::clamp(red,0.5f,1.f);
        blue = glm::clamp(blue,0.5f,1.f);
        green = glm::clamp(green,0.5f,1.f);
        

        color = {red,blue,green};
    }

} // namespace test
