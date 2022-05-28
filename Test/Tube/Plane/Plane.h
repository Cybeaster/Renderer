///////////////////////////////////////////////////////////////////////////////
// Plane.h
// =======
// class for a 3D plane with normal vector (a,b,c) and a point (x0,y0,z0)
// ax + by + cz + d = 0,  where d = -(ax0 + by0 + cz0)
//
// NOTE:
// 1. The default plane is z = 0 (a plane on XY axis)
// 2. The distance is the length from the origin to the plane
//
// Dependencies: Vector3, Line
//
//  AUTHOR: Song Ho Ahn (song.ahn@gmail.com)
// CREATED: 2016-01-19
// UPDATED: 2016-04-14
///////////////////////////////////////////////////////////////////////////////

#ifndef PLANE_H_DEF
#define PLANE_H_DEF

#include "Vectors.h"
#include "Line.h"

class Plane
{
public:
    // ctor/dtor
    Plane();
    Plane(float a, float b, float c, float d);          // 4 coeff of plane equation
    Plane(const Vector3& normal, const Vector3& point); // a point on the plane and normal vector
    ~Plane() {}

    // debug
    void printSelf() const;

    // setters/getters
    void set(float a, float b, float c, float d);
    void set(const Vector3& normal, const Vector3& point);  // set with  a point on the plane and normal
    const Vector3& getNormal() const { return normal; }
    float getD() const { return d; }                        // return 4th coefficient
    float getNormalLength() const { return normalLength; }  // return length of normal
    float getDistance() const { return distance; };         // return distance from the origin
    float getDistance(const Vector3& point);                // return distance from the point

    // convert plane equation with unit normal vector
    void normalize();

    // for intersection
    Vector3 intersect(const Line& line) const;              // intersect with a line
    Line intersect(const Plane& plane) const;               // intersect with another plane
    bool isIntersected(const Line& line) const;
    bool isIntersected(const Plane& plane) const;

protected:

private:
    Vector3 normal;     // normal vector of a plane
    float d;            // coefficient of constant term: d = -(a*x0 + b*y0 + c*z0)
    float normalLength; // length of normal vector
    float distance;     // distance from origin to plane
};

#endif
