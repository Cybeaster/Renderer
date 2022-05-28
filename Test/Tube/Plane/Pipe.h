///////////////////////////////////////////////////////////////////////////////
// Pipe.h
// ======
// base contour following a path
// The contour is a 2D shape on XY plane.
//
// Dependencies: Vector3, Plane, Line, Matrix4
//
//  AUTHOR: Song ho Ahn (song.ahn@gmail.com)
// CREATED: 2016-04-16
// UPDATED: 2016-04-29
///////////////////////////////////////////////////////////////////////////////

#ifndef PIPE_H_DEF
#define PIPE_H_DEF

#include <vector>
#include "Vectors.h"

class Pipe
{
public:
    // ctor/dtor
    Pipe();
    Pipe(const std::vector<Vector3>& pathPoints, const std::vector<Vector3>& contourPoints);
    ~Pipe() {}

    // setters/getters
    void set(const std::vector<Vector3>& pathPoints, const std::vector<Vector3>& contourPoints);
    void setPath(const std::vector<Vector3>& pathPoints);
    void setContour(const std::vector<Vector3>& contourPoints);
    void addPathPoint(const Vector3& point);

    int getPathCount() const                                        { return (int)path.size(); }
    const std::vector<Vector3>& getPathPoints() const               { return path; }
    const Vector3& getPathPoint(int index) const                    { return path.at(index); }
    int getContourCount() const                                     { return (int)contours.size(); }
    const std::vector< std::vector<Vector3> >& getContours() const  { return contours; }
    const std::vector<Vector3>& getContour(int index) const         { return contours.at(index); }
    const std::vector< std::vector<Vector3> >& getNormals() const   { return normals; }
    const std::vector<Vector3>& getNormal(int index) const          { return normals.at(index); }

protected:

private:
    // member functions
    void generateContours();
    void transformFirstContour();
    std::vector<Vector3> projectContour(int fromIndex, int toIndex);
    std::vector<Vector3> computeContourNormal(int pathIndex);

    std::vector<Vector3> path;
    std::vector<Vector3> contour;
    std::vector< std::vector<Vector3> > contours;
    std::vector< std::vector<Vector3> > normals;
};
#endif
