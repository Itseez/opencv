/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#ifndef OPENCV_CORE_OPENGL_HPP
#define OPENCV_CORE_OPENGL_HPP

#ifndef __cplusplus
#  error opengl.hpp header must be compiled as C++
#endif

#include "opencv2/core.hpp"
#include "ocl.hpp"

namespace cv { namespace ogl {

/** @addtogroup core_opengl
This section describes OpenGL interoperability.

To enable OpenGL support, configure OpenCV using CMake with WITH_OPENGL=ON . Currently OpenGL is
supported only with WIN32, GTK and Qt backends on Windows and Linux (MacOS and Android are not
supported). For GTK-2.0 backend gtkglext-1.0 library is required.

To use OpenGL functionality you should first create OpenGL context (window or frame buffer). You can
do this with namedWindow function or with other OpenGL toolkit (GLUT, for example).
*/
//! @{

/////////////////// OpenGL Objects ///////////////////

/** @brief Smart pointer for OpenGL buffer object with reference counting.

Buffer Objects are OpenGL objects that store an array of unformatted memory allocated by the OpenGL
context. These can be used to store vertex data, pixel data retrieved from images or the
framebuffer, and a variety of other things.

ogl::Buffer has interface similar with Mat interface and represents 2D array memory.

ogl::Buffer supports memory transfers between host and device and also can be mapped to CUDA memory.
 */
class CV_EXPORTS Buffer
{
public:
    /** @brief The target defines how you intend to use the buffer object.
    */
    enum Target
    {
        ARRAY_BUFFER         = 0x8892, //!< The buffer will be used as a source for vertex data
        ELEMENT_ARRAY_BUFFER = 0x8893, //!< The buffer will be used for indices (in glDrawElements, for example)
        PIXEL_PACK_BUFFER    = 0x88EB, //!< The buffer will be used for reading from OpenGL textures
        PIXEL_UNPACK_BUFFER  = 0x88EC  //!< The buffer will be used for writing to OpenGL textures
    };

    enum Access
    {
        READ_ONLY  = 0x88B8,
        WRITE_ONLY = 0x88B9,
        READ_WRITE = 0x88BA
    };

    /** @brief The constructors.

    Creates empty ogl::Buffer object, creates ogl::Buffer object from existed buffer ( abufId
    parameter), allocates memory for ogl::Buffer object or copies from host/device memory.
     */
    Buffer();

    /** @overload
    @param arows Number of rows in a 2D array.
    @param acols Number of columns in a 2D array.
    @param atype Array type ( CV_8UC1, ..., CV_64FC4 ). See Mat for details.
    @param abufId Buffer object name.
    @param autoRelease Auto release mode (if true, release will be called in object's destructor).
    */
    Buffer(int arows, int acols, int atype, unsigned int abufId, bool autoRelease = false);

    /** @overload
    @param asize 2D array size.
    @param atype Array type ( CV_8UC1, ..., CV_64FC4 ). See Mat for details.
    @param abufId Buffer object name.
    @param autoRelease Auto release mode (if true, release will be called in object's destructor).
    */
    Buffer(Size asize, int atype, unsigned int abufId, bool autoRelease = false);

    /** @overload
    @param arows Number of rows in a 2D array.
    @param acols Number of columns in a 2D array.
    @param atype Array type ( CV_8UC1, ..., CV_64FC4 ). See Mat for details.
    @param target Buffer usage. See cv::ogl::Buffer::Target .
    @param autoRelease Auto release mode (if true, release will be called in object's destructor).
    */
    Buffer(int arows, int acols, int atype, Target target = ARRAY_BUFFER, bool autoRelease = false);

    /** @overload
    @param asize 2D array size.
    @param atype Array type ( CV_8UC1, ..., CV_64FC4 ). See Mat for details.
    @param target Buffer usage. See cv::ogl::Buffer::Target .
    @param autoRelease Auto release mode (if true, release will be called in object's destructor).
    */
    Buffer(Size asize, int atype, Target target = ARRAY_BUFFER, bool autoRelease = false);

    /** @overload
    @param arr Input array (host or device memory, it can be Mat , cuda::GpuMat or std::vector ).
    @param target Buffer usage. See cv::ogl::Buffer::Target .
    @param autoRelease Auto release mode (if true, release will be called in object's destructor).
    */
    explicit Buffer(InputArray arr, Target target = ARRAY_BUFFER, bool autoRelease = false);

    /** @brief Allocates memory for ogl::Buffer object.

    @param arows Number of rows in a 2D array.
    @param acols Number of columns in a 2D array.
    @param atype Array type ( CV_8UC1, ..., CV_64FC4 ). See Mat for details.
    @param target Buffer usage. See cv::ogl::Buffer::Target .
    @param autoRelease Auto release mode (if true, release will be called in object's destructor).
     */
    void create(int arows, int acols, int atype, Target target = ARRAY_BUFFER, bool autoRelease = false);

    /** @overload
    @param asize 2D array size.
    @param atype Array type ( CV_8UC1, ..., CV_64FC4 ). See Mat for details.
    @param target Buffer usage. See cv::ogl::Buffer::Target .
    @param autoRelease Auto release mode (if true, release will be called in object's destructor).
    */
    void create(Size asize, int atype, Target target = ARRAY_BUFFER, bool autoRelease = false);

    /** @brief Decrements the reference counter and destroys the buffer object if needed.

    The function will call setAutoRelease(true) .
     */
    void release();

    /** @brief Sets auto release mode.

    The lifetime of the OpenGL object is tied to the lifetime of the context. If OpenGL context was
    bound to a window it could be released at any time (user can close a window). If object's destructor
    is called after destruction of the context it will cause an error. Thus ogl::Buffer doesn't destroy
    OpenGL object in destructor by default (all OpenGL resources will be released with OpenGL context).
    This function can force ogl::Buffer destructor to destroy OpenGL object.
    @param flag Auto release mode (if true, release will be called in object's destructor).
     */
    void setAutoRelease(bool flag);

    /** @brief Copies from host/device memory to OpenGL buffer.
    @param arr Input array (host or device memory, it can be Mat , cuda::GpuMat or std::vector ).
    @param target Buffer usage. See cv::ogl::Buffer::Target .
    @param autoRelease Auto release mode (if true, release will be called in object's destructor).
     */
    void copyFrom(InputArray arr, Target target = ARRAY_BUFFER, bool autoRelease = false);

    /** @overload */
    void copyFrom(InputArray arr, cuda::Stream& stream, Target target = ARRAY_BUFFER, bool autoRelease = false);

    /** @brief Copies from OpenGL buffer to host/device memory or another OpenGL buffer object.

    @param arr Destination array (host or device memory, can be Mat , cuda::GpuMat , std::vector or
    ogl::Buffer ).
     */
    void copyTo(OutputArray arr) const;

    /** @overload */
    void copyTo(OutputArray arr, cuda::Stream& stream) const;

    /** @brief Creates a full copy of the buffer object and the underlying data.

    @param target Buffer usage for destination buffer.
    @param autoRelease Auto release mode for destination buffer.
     */
    Buffer clone(Target target = ARRAY_BUFFER, bool autoRelease = false) const;

    /** @brief Binds OpenGL buffer to the specified buffer binding point.

    @param target Binding point. See cv::ogl::Buffer::Target .
     */
    void bind(Target target) const;

    /** @brief Unbind any buffers from the specified binding point.

    @param target Binding point. See cv::ogl::Buffer::Target .
     */
    static void unbind(Target target);

    /** @brief Maps OpenGL buffer to host memory.

    mapHost maps to the client's address space the entire data store of the buffer object. The data can
    then be directly read and/or written relative to the returned pointer, depending on the specified
    access policy.

    A mapped data store must be unmapped with ogl::Buffer::unmapHost before its buffer object is used.

    This operation can lead to memory transfers between host and device.

    Only one buffer object can be mapped at a time.
    @param access Access policy, indicating whether it will be possible to read from, write to, or both
    read from and write to the buffer object's mapped data store. The symbolic constant must be
    ogl::Buffer::READ_ONLY , ogl::Buffer::WRITE_ONLY or ogl::Buffer::READ_WRITE .
     */
    Mat mapHost(Access access);

    /** @brief Unmaps OpenGL buffer.
    */
    void unmapHost();

    //! map to device memory (blocking)
    cuda::GpuMat mapDevice();
    void unmapDevice();

    /** @brief Maps OpenGL buffer to CUDA device memory.

    This operation doesn't copy data. Several buffer objects can be mapped to CUDA memory at a time.

    A mapped data store must be unmapped with ogl::Buffer::unmapDevice before its buffer object is used.
     */
    cuda::GpuMat mapDevice(cuda::Stream& stream);

    /** @brief Unmaps OpenGL buffer.
    */
    void unmapDevice(cuda::Stream& stream);

    int rows() const;
    int cols() const;
    Size size() const;
    bool empty() const;

    int type() const;
    int depth() const;
    int channels() const;
    int elemSize() const;
    int elemSize1() const;

    //! get OpenGL opject id
    unsigned int bufId() const;

    class Impl;

private:
    Ptr<Impl> impl_;
    int rows_;
    int cols_;
    int type_;
};

class CV_EXPORTS VertexArray
{
public:
    VertexArray();

    VertexArray(unsigned int vertexArrayId, bool autoRelease = false);

    void bind() const;

    static void unbind();

    void setAutoRelease(bool flag);

    void vertexAttribPointer(InputArray arr, unsigned int index, int size, int stride, int offset) const;

    class Impl;

private:
    Ptr<Impl> impl_;
};

/** @brief Smart pointer for OpenGL 2D texture memory with reference counting.
 */
class CV_EXPORTS Texture2D
{
public:
    /** @brief An Image Format describes the way that the images in Textures store their data.
    */
    enum Format
    {
        NONE            = 0,
        DEPTH_COMPONENT = 0x1902, //!< Depth
        RGB             = 0x1907, //!< Red, Green, Blue
        RGBA            = 0x1908  //!< Red, Green, Blue, Alpha
    };

    /** @brief The constructors.

    Creates empty ogl::Texture2D object, allocates memory for ogl::Texture2D object or copies from
    host/device memory.
     */
    Texture2D();

    /** @overload */
    Texture2D(int arows, int acols, Format aformat, unsigned int atexId, bool autoRelease = false);

    /** @overload */
    Texture2D(Size asize, Format aformat, unsigned int atexId, bool autoRelease = false);

    /** @overload
    @param arows Number of rows.
    @param acols Number of columns.
    @param aformat Image format. See cv::ogl::Texture2D::Format .
    @param autoRelease Auto release mode (if true, release will be called in object's destructor).
    */
    Texture2D(int arows, int acols, Format aformat, bool autoRelease = false);

    /** @overload
    @param asize 2D array size.
    @param aformat Image format. See cv::ogl::Texture2D::Format .
    @param autoRelease Auto release mode (if true, release will be called in object's destructor).
    */
    Texture2D(Size asize, Format aformat, bool autoRelease = false);

    /** @overload
    @param arr Input array (host or device memory, it can be Mat , cuda::GpuMat or ogl::Buffer ).
    @param autoRelease Auto release mode (if true, release will be called in object's destructor).
    */
    explicit Texture2D(InputArray arr, bool autoRelease = false);

    /** @brief Allocates memory for ogl::Texture2D object.

    @param arows Number of rows.
    @param acols Number of columns.
    @param aformat Image format. See cv::ogl::Texture2D::Format .
    @param autoRelease Auto release mode (if true, release will be called in object's destructor).
     */
    void create(int arows, int acols, Format aformat, bool autoRelease = false);
    /** @overload
    @param asize 2D array size.
    @param aformat Image format. See cv::ogl::Texture2D::Format .
    @param autoRelease Auto release mode (if true, release will be called in object's destructor).
    */
    void create(Size asize, Format aformat, bool autoRelease = false);

    /** @brief Decrements the reference counter and destroys the texture object if needed.

    The function will call setAutoRelease(true) .
     */
    void release();

    /** @brief Sets auto release mode.

    @param flag Auto release mode (if true, release will be called in object's destructor).

    The lifetime of the OpenGL object is tied to the lifetime of the context. If OpenGL context was
    bound to a window it could be released at any time (user can close a window). If object's destructor
    is called after destruction of the context it will cause an error. Thus ogl::Texture2D doesn't
    destroy OpenGL object in destructor by default (all OpenGL resources will be released with OpenGL
    context). This function can force ogl::Texture2D destructor to destroy OpenGL object.
     */
    void setAutoRelease(bool flag);

    /** @brief Copies from host/device memory to OpenGL texture.

    @param arr Input array (host or device memory, it can be Mat , cuda::GpuMat or ogl::Buffer ).
    @param autoRelease Auto release mode (if true, release will be called in object's destructor).
     */
    void copyFrom(InputArray arr, bool autoRelease = false);

    /** @brief Copies from OpenGL texture to host/device memory or another OpenGL texture object.

    @param arr Destination array (host or device memory, can be Mat , cuda::GpuMat , ogl::Buffer or
    ogl::Texture2D ).
    @param ddepth Destination depth.
    @param autoRelease Auto release mode for destination buffer (if arr is OpenGL buffer or texture).
     */
    void copyTo(OutputArray arr, int ddepth = CV_32F, bool autoRelease = false) const;

    /** @brief Binds texture to current active texture unit for GL_TEXTURE_2D target.
    */
    void bind() const;

    int rows() const;
    int cols() const;
    Size size() const;
    bool empty() const;

    Format format() const;

    //! get OpenGL opject id
    unsigned int texId() const;

    class Impl;

private:
    Ptr<Impl> impl_;
    int rows_;
    int cols_;
    Format format_;
};

/** @brief Wrapper for OpenGL Client-Side Vertex arrays.

ogl::Arrays stores vertex data in ogl::Buffer objects.
 */
class CV_EXPORTS Arrays
{
public:
    /** @brief Default constructor
     */
    Arrays();

    /** @brief Sets an array of vertex coordinates.
    @param vertex array with vertex coordinates, can be both host and device memory.
    */
    void setVertexArray(InputArray vertex);

    /** @brief Resets vertex coordinates.
    */
    void resetVertexArray();

    /** @brief Sets an array of vertex colors.
    @param color array with vertex colors, can be both host and device memory.
     */
    void setColorArray(InputArray color);

    /** @brief Resets vertex colors.
    */
    void resetColorArray();

    /** @brief Sets an array of vertex normals.
    @param normal array with vertex normals, can be both host and device memory.
     */
    void setNormalArray(InputArray normal);

    /** @brief Resets vertex normals.
    */
    void resetNormalArray();

    /** @brief Sets an array of vertex texture coordinates.
    @param texCoord array with vertex texture coordinates, can be both host and device memory.
     */
    void setTexCoordArray(InputArray texCoord);

    /** @brief Resets vertex texture coordinates.
    */
    void resetTexCoordArray();

    /** @brief Releases all inner buffers.
    */
    void release();

    /** @brief Sets auto release mode all inner buffers.
    @param flag Auto release mode.
     */
    void setAutoRelease(bool flag);

    /** @brief Binds all vertex arrays.
    */
    void bind() const;

    /** @brief Returns the vertex count.
    */
    int size() const;
    bool empty() const;

private:
    int size_;
    Buffer vertex_;
    Buffer color_;
    Buffer normal_;
    Buffer texCoord_;
};


/** @brief Wrapper for OpenGL 3.2 Program object
 */
class CV_EXPORTS Program{
public:
    /** @brief Default constructor
     */
    Program();

    /** @brief Attach fragment and vertex shaders
    */
    void attachShaders(const std::string& fragment_shader_source, const std::string& vertex_shader_source);

    /** @brief Get the location of a specific uniform variable within a program object
    */
    int getUniformLocation(const std::string& name);

    /** @brief Get default fragment shader
    */
    static std::string getDefaultFragmentShader();

    /** @brief Get default vertex shader
    */
    static std::string getDefaultVertexShader();

    /** @brief Get the current program object
    */
    unsigned int getProgramId();

private:
    unsigned int program_;

};


/////////////////// Render Functions ///////////////////

//! render mode
enum RenderModes {
    POINTS         = 0x0000,
    LINES          = 0x0001,
    LINE_LOOP      = 0x0002,
    LINE_STRIP     = 0x0003,
    TRIANGLES      = 0x0004,
    TRIANGLE_STRIP = 0x0005,
    TRIANGLE_FAN   = 0x0006,
    QUADS          = 0x0007,
    QUAD_STRIP     = 0x0008,
    POLYGON        = 0x0009
};

/** @brief Render OpenGL texture or primitives.
@param tex Texture to draw.
@param wndRect Region of window, where to draw a texture (normalized coordinates).
@param texRect Region of texture to draw (normalized coordinates).
 */
CV_EXPORTS void render(const Texture2D& tex,
    Rect_<double> wndRect = Rect_<double>(0.0, 0.0, 1.0, 1.0),
    Rect_<double> texRect = Rect_<double>(0.0, 0.0, 1.0, 1.0));

/** @overload
@param arr Array of privitives vertices.
@param mode Render mode. One of cv::ogl::RenderModes
@param color Color for all vertices. Will be used if arr doesn't contain color array.
*/
CV_EXPORTS void render(const Arrays& arr, int mode = POINTS, Scalar color = Scalar::all(255));

/** @overload
@param arr Array of privitives vertices.
@param indices Array of vertices indices (host or device memory).
@param mode Render mode. One of cv::ogl::RenderModes
@param color Color for all vertices. Will be used if arr doesn't contain color array.
*/
CV_EXPORTS void render(const Arrays& arr, InputArray indices, int mode = POINTS, Scalar color = Scalar::all(255));

/////////////////// Export to Graphic device ///////////////////

CV_EXPORTS void uniformMatrix4fv(int location, int count, float* value);

/////////////////// CL-GL Interoperability Functions ///////////////////

namespace ocl {
using namespace cv::ocl;

// TODO static functions in the Context class
/** @brief Creates OpenCL context from GL.
@return Returns reference to OpenCL Context
 */
CV_EXPORTS Context& initializeContextFromGL();

} // namespace cv::ogl::ocl

/** @brief Converts InputArray to Texture2D object.
@param src     - source InputArray.
@param texture - destination Texture2D object.
 */
CV_EXPORTS void convertToGLTexture2D(InputArray src, Texture2D& texture);

/** @brief Converts Texture2D object to OutputArray.
@param texture - source Texture2D object.
@param dst     - destination OutputArray.
 */
CV_EXPORTS void convertFromGLTexture2D(const Texture2D& texture, OutputArray dst);

/** @brief Maps Buffer object to process on CL side (convert to UMat).

Function creates CL buffer from GL one, and then constructs UMat that can be used
to process buffer data with OpenCV functions. Note that in current implementation
UMat constructed this way doesn't own corresponding GL buffer object, so it is
the user responsibility to close down CL/GL buffers relationships by explicitly
calling unmapGLBuffer() function.
@param buffer      - source Buffer object.
@param accessFlags - data access flags (ACCESS_READ|ACCESS_WRITE).
@return Returns UMat object
 */
CV_EXPORTS UMat mapGLBuffer(const Buffer& buffer, AccessFlag accessFlags = ACCESS_READ | ACCESS_WRITE);

/** @brief Unmaps Buffer object (releases UMat, previously mapped from Buffer).

Function must be called explicitly by the user for each UMat previously constructed
by the call to mapGLBuffer() function.
@param u           - source UMat, created by mapGLBuffer().
 */
CV_EXPORTS void unmapGLBuffer(UMat& u);

//! @}
}} // namespace cv::ogl

namespace cv { namespace cuda {

/** @brief Sets a CUDA device and initializes it for the current thread with OpenGL interoperability.

This function should be explicitly called after OpenGL context creation and before any CUDA calls.
@param device System index of a CUDA device starting with 0.
@ingroup core_opengl
 */
CV_EXPORTS void setGlDevice(int device = 0);

}}

//! @cond IGNORED

////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////

inline
cv::ogl::Buffer::Buffer(int arows, int acols, int atype, Target target, bool autoRelease) : rows_(0), cols_(0), type_(0)
{
    create(arows, acols, atype, target, autoRelease);
}

inline
cv::ogl::Buffer::Buffer(Size asize, int atype, Target target, bool autoRelease) : rows_(0), cols_(0), type_(0)
{
    create(asize, atype, target, autoRelease);
}

inline
void cv::ogl::Buffer::create(Size asize, int atype, Target target, bool autoRelease)
{
    create(asize.height, asize.width, atype, target, autoRelease);
}

inline
int cv::ogl::Buffer::rows() const
{
    return rows_;
}

inline
int cv::ogl::Buffer::cols() const
{
    return cols_;
}

inline
cv::Size cv::ogl::Buffer::size() const
{
    return Size(cols_, rows_);
}

inline
bool cv::ogl::Buffer::empty() const
{
    return rows_ == 0 || cols_ == 0;
}

inline
int cv::ogl::Buffer::type() const
{
    return type_;
}

inline
int cv::ogl::Buffer::depth() const
{
    return CV_MAT_DEPTH(type_);
}

inline
int cv::ogl::Buffer::channels() const
{
    return CV_MAT_CN(type_);
}

inline
int cv::ogl::Buffer::elemSize() const
{
    return CV_ELEM_SIZE(type_);
}

inline
int cv::ogl::Buffer::elemSize1() const
{
    return CV_ELEM_SIZE1(type_);
}

///////

inline
cv::ogl::Texture2D::Texture2D(int arows, int acols, Format aformat, bool autoRelease) : rows_(0), cols_(0), format_(NONE)
{
    create(arows, acols, aformat, autoRelease);
}

inline
cv::ogl::Texture2D::Texture2D(Size asize, Format aformat, bool autoRelease) : rows_(0), cols_(0), format_(NONE)
{
    create(asize, aformat, autoRelease);
}

inline
void cv::ogl::Texture2D::create(Size asize, Format aformat, bool autoRelease)
{
    create(asize.height, asize.width, aformat, autoRelease);
}

inline
int cv::ogl::Texture2D::rows() const
{
    return rows_;
}

inline
int cv::ogl::Texture2D::cols() const
{
    return cols_;
}

inline
cv::Size cv::ogl::Texture2D::size() const
{
    return Size(cols_, rows_);
}

inline
bool cv::ogl::Texture2D::empty() const
{
    return rows_ == 0 || cols_ == 0;
}

inline
cv::ogl::Texture2D::Format cv::ogl::Texture2D::format() const
{
    return format_;
}

///////

// WARNING: unreachable code using Ninja
#if defined _MSC_VER && _MSC_VER >= 1920
#pragma warning(push)
#pragma warning(disable: 4702)
#endif
inline
cv::ogl::Arrays::Arrays() : size_(0)
{
}
#if defined _MSC_VER && _MSC_VER >= 1920
#pragma warning(pop)
#endif

inline
int cv::ogl::Arrays::size() const
{
    return size_;
}

inline
bool cv::ogl::Arrays::empty() const
{
    return size_ == 0;
}

//! @endcond

#endif /* OPENCV_CORE_OPENGL_HPP */
