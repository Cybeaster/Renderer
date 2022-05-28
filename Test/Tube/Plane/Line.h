///////////////////////////////////////////////////////////////////////////////
// Line.h
// ======
// class to construct a line with parametric form
// Line = p + aV (a point and a direction vector on the line)
//
// Dependency: Vector2, Vector3
//
//  AUTHOR: Song Ho Ahn (song.ahn@gmail.com)
// CREATED: 2015-12-18
// UPDATED: 2020-07-17
///////////////////////////////////////////////////////////////////////////////

#ifndef LINE_H_DEF
#define LINE_H_DEF

#include <cmath>
#include "Vectors.h"



class Line
{
public:
    // ctor/dtor
    Line() : direction(Vector3(0,0,0)), point(Vector3(0,0,0)) {}
    Line(const Vector3& v, const Vector3& p) : direction(v), point(p) {}    // with 3D direction and a point
    Line(const Vector2& v, const Vector2& p);                               // with 2D direction and a point
    Line(float slope, float intercept);                                     // with 2D slope-intercept form
    ~Line() {};

    // getters/setters
    void set(const Vector3& v, const Vector3& p);               // from 3D
    void set(const Vector2& v, const Vector2& p);               // from 2D
    void set(float slope, float intercept);                     // from slope-intercept form
    void setPoint(Vector3& p)           { point = p; }
    void setDirection(const Vector3& v) { direction = v; }
    const Vector3& getPoint() const     { return point; }
    const Vector3& getDirection() const { return direction; }
    void printSelf();

    // find intersect point with other line
    Vector3 intersect(const Line& line);
    bool isIntersected(const Line& line);

protected:

private:
    Vector3 direction;
    Vector3 point;
};

#endif

