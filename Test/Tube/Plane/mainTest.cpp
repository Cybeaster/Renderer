

    
    ///////////////////////////////////////////////////////////////////////////////
    // main.cpp
    // ========
    // spiral pipe
    //
    //  AUTHOR: Song Ho Ahn (song.ahn@gmail.com)
    // CREATED: 2016-04-12
    // UPDATED: 2020-02-29
    ///////////////////////////////////////////////////////////////////////////////

    #ifdef __APPLE__
    #include <GLUT/glut.h>
    #else
    #include <GL/glew.h>
    #endif

    #include <cstdlib>
    #include <iostream>
    #include <sstream>
    #include <string>
    #include <iomanip>
    #include <vector>
    #include "Vectors.h"
    #include "Matrices.h"
    #include "Plane.h"
    #include "Line.h"
    #include "Pipe.h"


    // GLUT CALLBACK functions ////////////////////////////////////////////////////
    void displayCB();
    void reshapeCB(int w, int h);
    void timerCB(int millisec);
    void idleCB();
    void keyboardCB(unsigned char key, int x, int y);
    void mouseCB(int button, int stat, int x, int y);
    void mouseMotionCB(int x, int y);
    void mousePassiveMotionCB(int x, int y);

    // CALLBACK function when exit() called ///////////////////////////////////////
    void exitCB();

    // function declearations /////////////////////////////////////////////////////
    void initGL();
    int  initGLUT(int argc, char **argv);
    bool initSharedMem();
    void clearSharedMem();
    void initLights();
    void setCamera(float posX, float posY, float posZ, float targetX, float targetY, float targetZ);
    void drawString(const char *str, int x, int y, float color[4], void *font);
    void drawString3D(const char *str, float pos[3], float color[4], void *font);
    void showInfo();
    void drawPipe();
    void drawPath();
    void draw();

    std::vector<Vector3> buildSpiralPath(float r1, float r2, float h1, float h2, float turns, int points);
    std::vector<Vector3> buildCircle(float radius, int steps);



    // constants
    const int SCREEN_WIDTH = 600;
    const int SCREEN_HEIGHT = 600;
    const int CIRCLE_SECTORS = 48;

 
    bool mouseLeftDown;
    bool mouseRightDown;
    float mouseX, mouseY;
    float cameraAngleX;
    float cameraAngleY;
    float cameraDistance = 10;
    int screenWidth, screenHeight;
    int drawMode;
    int currIndex = 0;
    bool animating = true;
    std::vector<Vector3> path;
    std::vector<Vector3> circle;
    Pipe pipe;


    ///////////////////////////////////////////////////////////////////////////////
    // draw a pipe
    ///////////////////////////////////////////////////////////////////////////////
    void drawPipe()
    {
        if(drawMode == 0)
        {
            glColor4f(1, 1, 0, 1);
        }
        else
        {
            glColor4f(1, 1, 0, 0.3f);
        }

        glLineWidth(1);
        int count = pipe.getContourCount();
        for(int i = 0; i < count; ++i)
        {
            std::vector<Vector3> contour = pipe.getContour(i);
            std::vector<Vector3> normal = pipe.getNormal(i);
            glBegin(GL_LINES);
            for(int j = 0; j < (int)contour.size() - 1; ++j)
            {
                glNormal3fv(&normal[j].x);
                glVertex3fv(&contour[j].x);
                glNormal3fv(&normal[j+1].x);
                glVertex3fv(&contour[j+1].x);
            }
            glEnd();
        }

        // surface
        for(int i = 0; i < count - 1; ++i)
        {
            std::vector<Vector3> c1 = pipe.getContour(i);
            std::vector<Vector3> c2 = pipe.getContour(i+1);
            std::vector<Vector3> n1 = pipe.getNormal(i);
            std::vector<Vector3> n2 = pipe.getNormal(i+1);
            glBegin(GL_TRIANGLE_STRIP);
            for(int j = 0; j < (int)c2.size(); ++j)
            {
                glNormal3fv(&n2[j].x);
                glVertex3fv(&c2[j].x);
                glNormal3fv(&n1[j].x);
                glVertex3fv(&c1[j].x);
            }
            glEnd();
        }
    }



    ///////////////////////////////////////////////////////////////////////////////
    // draw lines along the path
    ///////////////////////////////////////////////////////////////////////////////
    void drawPath()
    {
        glDisable(GL_LIGHTING);
        glDisable(GL_DEPTH_TEST);

        // lines
        glColor3f(1.0f, 0.5f, 0.0f);
        glLineWidth(2.0f);
        glBegin(GL_LINES);

        int count = pipe.getPathCount();
        for(int i = 0; i < count-1; ++i)
        {
            glVertex3fv(&pipe.getPathPoint(i).x);
            glVertex3fv(&pipe.getPathPoint(i+1).x);
        }
        glEnd();

        // points
        glColor3f(0.0f, 1.0f, 1.0f);
        glPointSize(5.0f);
        glBegin(GL_POINTS);
        for(int i = 0; i < count; ++i)
        {
            glVertex3fv(&pipe.getPathPoint(i).x);
        }
        glEnd();
        glPointSize(1); // reset

        glEnable(GL_DEPTH_TEST);
        glEnable(GL_LIGHTING);
    }



    ///////////////////////////////////////////////////////////////////////////////
    // draw a grid on the xy plane
    ///////////////////////////////////////////////////////////////////////////////
    void drawGrid(float size, float step)
    {
        glDisable(GL_LIGHTING);
        glDisable(GL_DEPTH_TEST);
        glLineWidth(0.5f);

        glBegin(GL_LINES);

        glColor4f(0.5f, 0.5f, 0.5f, 0.5f);
        for(float i=step; i <= size; i+= step)
        {
            glVertex3f(-size,  i, 0);   // lines parallel to X-axis
            glVertex3f( size,  i, 0);
            glVertex3f(-size, -i, 0);   // lines parallel to X-axis
            glVertex3f( size, -i, 0);

            glVertex3f( i, -size, 0);   // lines parallel to Y-axis
            glVertex3f( i,  size, 0);
            glVertex3f(-i, -size, 0);   // lines parallel to Y-axis
            glVertex3f(-i,  size, 0);

            glVertex3f(-size,  0,  i);   // lines parallel to X-axis
            glVertex3f( size,  0,  i);
            glVertex3f(-size,  0, -i);   // lines parallel to X-axis
            glVertex3f( size,  0, -i);

            glVertex3f( i, 0, -size);   // lines parallel to Z-axis
            glVertex3f( i, 0,  size);
            glVertex3f(-i, 0, -size);   // lines parallel to Z-axis
            glVertex3f(-i, 0,  size);
    }

        // x-axis
        glColor4f(1.0f, 0, 0, 0.5f);
        glVertex3f(-size, 0, 0);
        glVertex3f( size, 0, 0);

        // y-axis
        glColor4f(0, 1.0f, 0, 0.5f);
        glVertex3f(0, -size, 0);
        glVertex3f(0,  size, 0);

        // z-axis
        glColor4f(0, 0, 1.0f, 0.5f);
        glVertex3f(0, 0, -size);
        glVertex3f(0, 0,  size);

        glEnd();

        glLineWidth(1.0f);
        glEnable(GL_DEPTH_TEST);
        glEnable(GL_LIGHTING);
    }



    ///////////////////////////////////////////////////////////////////////////////
    // draw 3D
    ///////////////////////////////////////////////////////////////////////////////
    void draw()
    {
        //drawGrid(10, 1);
        drawPath();
        drawPipe();
    }








    ///////////////////////////////////////////////////////////////////////////////
    // initialize global variables
    ///////////////////////////////////////////////////////////////////////////////
    bool initSharedMem()
    {
        screenWidth = SCREEN_WIDTH;
        screenHeight = SCREEN_HEIGHT;
        mouseLeftDown = mouseRightDown = false;
        cameraAngleX = cameraAngleY = 0;
        drawMode = 0;

        // generate path for pipe
        path = buildSpiralPath(4, 1, -3, 3, 3.5, 200);
        std::cout << "fitst point: " << path[0] << std::endl;
        std::cout << "last point: " << path[path.size()-1] << std::endl;

        // sectional contour of pipe
        circle = buildCircle(0.5f, CIRCLE_SECTORS); // radius, segments

        // configure pipe
        std::vector<Vector3> p(1, path[0]);
        pipe.set(p, circle);
        currIndex = 0;

        return true;
    }




    ///////////////////////////////////////////////////////////////////////////////
    // initialize lights
    ///////////////////////////////////////////////////////////////////////////////
    void initLights()
    {
        // set up light colors (ambient, diffuse, specular)
        GLfloat lightKa[] = {0.0f, 0.0f, 0.0f, 1.0f};  // ambient light
        GLfloat lightKd[] = {1.0f, 1.0f, 1.0f, 1.0f};  // diffuse light
        GLfloat lightKs[] = {1, 1, 1, 1};           // specular light
        glLightfv(GL_LIGHT0, GL_AMBIENT, lightKa);
        glLightfv(GL_LIGHT0, GL_DIFFUSE, lightKd);
        glLightfv(GL_LIGHT0, GL_SPECULAR, lightKs);

        // position the light
        float lightPos[4] = {0, 0, 1, 0}; // positional light
        glLightfv(GL_LIGHT0, GL_POSITION, lightPos);

        glEnable(GL_LIGHT0);                        // MUST enable each light source after configuration
    }




    ///////////////////////////////////////////////////////////////////////////////
    // generate a spiral path along y-axis
    // r1: starting radius
    // r2: ending radius
    // h1: starting height
    // h2: ending height
    // turns: # of revolutions
    // points: # of points
    std::vector<Vector3> buildSpiralPath(float r1, float r2, float h1, float h2,
                                        float turns, int points)
    {
        const float PI = acos(-1);
        std::vector<Vector3> vertices;
        Vector3 vertex;
        float r = r1;
        float rStep = (r2 - r1) / (points - 1);
        float y = h1;
        float yStep = (h2 - h1) / (points - 1);
        float a = 0;
        float aStep = (turns * 2 * PI) / (points - 1);
        for(int i = 0; i < points; ++i)
        {
            vertex.x = r * cos(a);
            vertex.z = r * sin(a);
            vertex.y = y;
            vertices.push_back(vertex);
            // next
            r += rStep;
            y += yStep;
            a += aStep;
        }
        return vertices;
    }



    ///////////////////////////////////////////////////////////////////////////////
    // generate points of a circle on xy plane
    ///////////////////////////////////////////////////////////////////////////////
    std::vector<Vector3> buildCircle(float radius, int steps)
    {
        std::vector<Vector3> points;
        if(steps < 2) return points;

        const float PI2 = acos(-1) * 2.0f;
        float x, y, a;
        for(int i = 0; i <= steps; ++i)
        {
            a = PI2 / steps * i;
            x = radius * cosf(a);
            y = radius * sinf(a);
            points.push_back(Vector3(x, y, 0));
        }
        return points;
    }






    //=============================================================================
    // CALLBACKS
    //=============================================================================

    void displayCB()
    {
        // add point
        if(animating)
        {
            ++currIndex;
            if(currIndex < path.size())
            {
                pipe.addPathPoint(path[currIndex]);
            }
            else
            {
                std::vector<Vector3> p(1, path[0]);
                pipe.set(p, circle);
                currIndex = 0;
            }
        }

        // clear framebuffer
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

  

        // tramsform camera
        glTranslatef(0, 0, -cameraDistance);
        glRotatef(cameraAngleX, 1, 0, 0);   // pitch
        glRotatef(cameraAngleY, 0, 1, 0);   // heading

        // draw 3D
        draw();

        showInfo();

        glPopMatrix();
        
    }


    void reshapeCB(int width, int height)
    {
        screenWidth = width;
        screenHeight = height;

        // set viewport to be the entire window
        glViewport(0, 0, (GLsizei)width, (GLsizei)height);

        // set perspective viewing frustum
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        gluPerspective(60.0f, (float)(width)/height, 1.0f, 1000.0f); // FOV, AspectRatio, NearClip, FarClip

        // switch to modelview matrix in order to set scene
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
    }

