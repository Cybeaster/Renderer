///////////////////////////////////////////////////////////////////////////////
// Line.cpp
// ========
// class to construct a line with parametric form
// Line = p + aV (a point and a direction vector on the line)
//
// Dependency: Vector2, Vector3
//
//  AUTHOR: Song Ho Ahn (song.ahn@gmail.com)
// CREATED: 2015-12-18
// UPDATED: 2020-07-17
///////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include "Line.h"


///////////////////////////////////////////////////////////////////////////////
// ctor
// convert 2D slope-intercept form to parametric form
///////////////////////////////////////////////////////////////////////////////
Line::Line(float slope, float intercept)
{
    set(slope, intercept);
}



///////////////////////////////////////////////////////////////////////////////
// ctor with 2D direction and point
///////////////////////////////////////////////////////////////////////////////
Line::Line(const Vector2& direction, const Vector2& point)
{
    set(direction, point);
}



///////////////////////////////////////////////////////////////////////////////
// setters
///////////////////////////////////////////////////////////////////////////////
void Line::set(const Vector3& v, const Vector3& p)
{
    this->direction = v;
    this->point = p;
}

void Line::set(const Vector2& v, const Vector2& p)
{
    // convert 2D to 3D
    this->direction = Vector3(v.x, v.y, 0);
    this->point = Vector3(p.x, p.y, 0);
}

void Line::set(float slope, float intercept)
{
    // convert slope-intercept form (2D) to parametric form (3D)
    this->direction = Vector3(1, slope, 0);
    this->point = Vector3(0, intercept, 0);
}



///////////////////////////////////////////////////////////////////////////////
// debug
///////////////////////////////////////////////////////////////////////////////
void Line::printSelf()
{
    std::cout << "Line\n"
              << "====\n"
              << "Direction: " << this->direction << "\n"
              << "    Point: " << this->point << std::endl;
}



///////////////////////////////////////////////////////////////////////////////
// find the intersection point with the other line.
// If no intersection, return a point with NaN in it.
//
// Line1 = p1 + aV1 (this)
// Line2 = p2 + bV2 (other)
//
// Intersect:
// p1 + aV1 = p2 + bV2
//      aV1 = (p2-p1) + bV2
//   aV1xV2 = (p2-p1)xV2
//        a = (p2-p1)xV2 / (V1xV2)
//        a = ((p2-p1)xV2).(V1xV2) / (V1xV2).(V1xV2)
///////////////////////////////////////////////////////////////////////////////
Vector3 Line::intersect(const Line& line)
{
    const Vector3 v2 = line.getDirection();
    const Vector3 p2 = line.getPoint();
    Vector3 result = Vector3(NAN, NAN, NAN);    // default with NaN

    // find v3 = (p2 - p1) x V2
    Vector3 v3 = (p2 - point).cross(v2);

    // find v4 = V1 x V2
    Vector3 v4 = direction.cross(v2);

    // find (V1xV2) . (V1xV2)
    float dot = v4.dot(v4);

    // if both V1 and V2 are same direction, return NaN point
    if(dot == 0)
        return result;

    // find a = ((p2-p1)xV2).(V1xV2) / (V1xV2).(V1xV2)
    float alpha = v3.dot(v4) / dot;

    /*
    // if both V1 and V2 are same direction, return NaN point
    if(v4.x == 0 && v4.y == 0 && v4.z == 0)
        return result;

    float alpha = 0;
    if(v4.x != 0)
        alpha = v3.x / v4.x;
    else if(v4.y != 0)
        alpha = v3.y / v4.y;
    else if(v4.z != 0)
        alpha = v3.z / v4.z;
    else
        return result;
    */

    // find intersect point
    result = point + (alpha * direction);
    return result;
}



///////////////////////////////////////////////////////////////////////////////
// determine if it intersects with the other line
///////////////////////////////////////////////////////////////////////////////
bool Line::isIntersected(const Line& line)
{
    // if 2 lines are same direction, the magnitude of cross product is 0
    Vector3 v = this->direction.cross(line.getDirection());
    if(v.x == 0 && v.y == 0 && v.z == 0)
        return false;
    else
        return true;
}
