#include <iostream>

#ifdef _WIN32
    #define WIN32_LEAN_AND_MEAN 1
    #define NOMINMAX 1
    #include <windows.h>
#endif

#if defined(__APPLE__)
    #include <OpenGL/gl.h>
    #include <OpenGL/glu.h>
#else
    #include <GL/gl.h>
    #include <GL/glu.h>
#endif

#include "opencv2/core.hpp"
#include "opencv2/core/opengl.hpp"
#include "opencv2/core/cuda.hpp"
#include "opencv2/highgui.hpp"

using namespace std;
using namespace cv;
using namespace cv::cuda;

const int win_width = 800;
const int win_height = 640;

struct DrawData
{
    ogl::VertexArray vao;
    ogl::Buffer vbo;
    ogl::Program program;
    ogl::Texture2D tex;
};

static cv::Mat rot(float angle)
{
    cv::Mat R_y = (cv::Mat_<float>(4,4) <<
        cos(angle), 0, sin(angle), 0,
        0, 1, 0, 0,
        -sin(angle), 0, cos(angle), 0,
        0, 0, 0, 1);

    return R_y;
}

static void draw(void* userdata) {
    DrawData* data = static_cast<DrawData*>(userdata);
    static float angle = 0.0f;
    angle += 1.f;

    cv::Mat trans = rot(CV_PI * angle / 360.f);

    glClearColor(0.0, 0.0, 0.0, 1.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    ogl::uniformMatrix4fv(data->program.getUniformLocation("transform"), 1, trans.ptr<float>());

    data->tex.bind();
    data->vao.bind();
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
}

int main(int argc, char* argv[])
{
    string filename;
    if (argc < 2)
    {
        cout << "Usage: " << argv[0] << " image" << endl;
        filename = "baboon.jpg";
    }
    else
        filename = argv[1];

    Mat img = imread(samples::findFile(filename));
    if (img.empty())
    {
        cerr << "Can't open image " << filename << endl;
        return -1;
    }
    flip(img, img, 0);

    namedWindow("OpenGL", WINDOW_OPENGL);
    resizeWindow("OpenGL", win_width, win_height);

    DrawData data;
    data.program.attachShaders(ogl::Program::getDefaultFragmentShader(), ogl::Program::getDefaultVertexShader()); // new

    std::vector<float> vertex = {
        // Positions         // Texture Coords
         1.0f,  1.0f,  0.0f,  1.0f,  1.0f, // Top Right
         1.0f, -1.0f,  0.0f,  1.0f,  0.0f, // Bottom Right
        -1.0f,  1.0f,  0.0f,  0.0f,  1.0f, // Top Left
        -1.0f, -1.0f,  0.0f,  0.0f,  0.0f  // Bottom Left
    };
    data.vbo.copyFrom(vertex);
    data.vbo.bind(cv::ogl::Buffer::ARRAY_BUFFER);
    data.vao = ogl::VertexArray(0, false);
    data.vao.vertexAttribPointer(vertex, 0, 3, 5, 0);
    data.vao.vertexAttribPointer(vertex, 1, 2, 5, 3);

    data.tex.copyFrom(img);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

    setOpenGlDrawCallback("OpenGL", draw, &data);

    for (;;)
    {
        updateWindow("OpenGL");
        char key = (char)waitKey(40);
        if (key == 27)
            break;
    }

    setOpenGlDrawCallback("OpenGL", 0, 0);
    destroyAllWindows();
    return 0;
}
