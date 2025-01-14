/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
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
//   * The name of Intel Corporation may not be used to endorse or promote products
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

#include "precomp.hpp"
#include "backend.hpp"

#if defined (HAVE_GTK)

#include <gtk/gtk.h>

#if (GTK_MAJOR_VERSION == 2) && defined(HAVE_OPENGL) && !defined(HAVE_GTKGLEXT)
  #undef HAVE_OPENGL  // gtkglext is required
#endif

#include <gdk/gdkkeysyms.h>
#include <gdk-pixbuf/gdk-pixbuf.h>
#include <stdio.h>

#if (GTK_MAJOR_VERSION == 2)
  #define GTK_VERSION2 1
#endif //GTK_MAJOR_VERSION >= 2.0
#if (GTK_MAJOR_VERSION == 3)
  #define GTK_VERSION3 1
#endif //GTK_MAJOR_VERSION >= 3
#if (GTK_MAJOR_VERSION == 3 && GTK_MINOR_VERSION >= 4)
  #define GTK_VERSION3_4 1
#endif

#ifdef HAVE_OPENGL
  #ifdef GTK_VERSION3
    #include <gtk/gtkglarea.h>
  #else
    #include <gtk/gtkgl.h>
    #include <GL/glu.h>
  #endif
  #include <GL/gl.h>
#endif

#include <opencv2/core/utils/logger.hpp>
#include "opencv2/core/utils/trace.hpp"
#include "opencv2/imgproc.hpp"

using namespace cv;

#ifndef BIT_ALLIN
    #define BIT_ALLIN(x,y) ( ((x)&(y)) == (y) )
#endif
#ifndef BIT_MAP
    #define BIT_MAP(x,y,z) ( ((x)&(y)) ? (z) : 0 )
#endif

// TODO Fix the initial window size when flags=0.  Right now the initial window is by default
// 320x240 size.  A better default would be actual size of the image.  Problem
// is determining desired window size with trackbars while still allowing resizing.
//
// Gnome Totem source may be of use here, see bacon_video_widget_set_scale_ratio
// in totem/src/backend/bacon-video-widget-xine.c

////////////////////////////////////////////////////////////
// CvImageWidget GTK Widget Public API
////////////////////////////////////////////////////////////

struct CvImageWidget
{
    GtkWidget widget;
    Mat original_image;
    Mat scaled_image;
    int flags;
};

struct CvImageWidgetClass
{
  GtkWidgetClass parent_class;
};


/** Allocate new image viewer widget */
GtkWidget*     cvImageWidgetNew      (int flags);

// standard GTK object macros
#define CV_IMAGE_WIDGET(obj)          G_TYPE_CHECK_INSTANCE_CAST (obj, cvImageWidget_get_type (), CvImageWidget)
#define CV_IMAGE_WIDGET_CLASS(klass)  GTK_CHECK_CLASS_CAST (klass, cvImageWidget_get_type (), CvImageWidgetClass)
#define CV_IS_IMAGE_WIDGET(obj)       G_TYPE_CHECK_INSTANCE_TYPE (obj, cvImageWidget_get_type ())

/////////////////////////////////////////////////////////////////////////////
// Private API ////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////
GType        cvImageWidget_get_type (void);

static GtkWidgetClass * parent_class = NULL;

// flag to help size initial window
#define CV_WINDOW_NO_IMAGE 2

/** Set the image to display in the widget */
static
void cvImageWidgetSetImage(CvImageWidget * widget, InputArray arr)
{
    Mat mat = arr.getMat();

    if(!widget->original_image.empty() && mat.size() != widget->original_image.size()){
        widget->original_image.release();
    }
    if(widget->original_image.empty()){
        widget->original_image.create(mat.size(), CV_8UC3);
        gtk_widget_queue_resize( GTK_WIDGET( widget ) );
    }
    convertToShow(arr.getMat(), widget->original_image);
    if(!widget->scaled_image.empty()){
        cv::resize(widget->original_image, widget->scaled_image, widget->scaled_image.size(), 0, 0, cv::INTER_AREA );
    }

    // window does not refresh without this
    gtk_widget_queue_draw( GTK_WIDGET(widget) );
}

GtkWidget*
cvImageWidgetNew (int flags)
{
  CvImageWidget *image_widget;

  image_widget = CV_IMAGE_WIDGET( gtk_widget_new (cvImageWidget_get_type (), NULL) );
  CV_Assert(image_widget && "GTK widget creation is failed. Ensure that there is no GTK2/GTK3 libraries conflict");
  image_widget->original_image = 0;
  image_widget->scaled_image = 0;
  image_widget->flags = flags | CV_WINDOW_NO_IMAGE;

  return GTK_WIDGET (image_widget);
}

static void
cvImageWidget_realize (GtkWidget *widget)
{
  GdkWindowAttr attributes;
  gint attributes_mask;

#if defined(GTK_VERSION3)
  GtkAllocation allocation;
  gtk_widget_get_allocation(widget, &allocation);
#endif //GTK_VERSION3

  //printf("cvImageWidget_realize\n");
  g_return_if_fail (widget != NULL);
  g_return_if_fail (CV_IS_IMAGE_WIDGET (widget));

  gtk_widget_set_realized(widget, TRUE);

#if defined(GTK_VERSION3)
  attributes.x = allocation.x;
  attributes.y = allocation.y;
  attributes.width = allocation.width;
  attributes.height = allocation.height;
#elif defined(GTK_VERSION2)
  attributes.x = widget->allocation.x;
  attributes.y = widget->allocation.y;
  attributes.width = widget->allocation.width;
  attributes.height = widget->allocation.height;
#endif //GTK_VERSION3

  attributes.wclass = GDK_INPUT_OUTPUT;
  attributes.window_type = GDK_WINDOW_CHILD;
  attributes.event_mask = gtk_widget_get_events (widget) |
    GDK_EXPOSURE_MASK | GDK_BUTTON_PRESS_MASK |
    GDK_BUTTON_RELEASE_MASK | GDK_POINTER_MOTION_MASK;
  attributes.visual = gtk_widget_get_visual (widget);

#if defined(GTK_VERSION3)
  attributes_mask = GDK_WA_X | GDK_WA_Y | GDK_WA_VISUAL;
  gtk_widget_set_window(
      widget,
      gdk_window_new(
        gtk_widget_get_parent_window(widget),
        &attributes,
        attributes_mask
        )
      );

  gtk_widget_set_style(
      widget,
      gtk_style_attach(
        gtk_widget_get_style(widget),
        gtk_widget_get_window(widget)
        )
      );

  gdk_window_set_user_data (
      gtk_widget_get_window(widget),
      widget
      );

  gtk_style_set_background (
      gtk_widget_get_style(widget),
      gtk_widget_get_window(widget),
      GTK_STATE_ACTIVE
      );
 #elif defined(GTK_VERSION2)
  // The following lines are included to prevent breaking
  // compatibility with older Gtk2 (<gtk+-2.18) libraries.
  attributes.colormap = gtk_widget_get_colormap (widget);
  attributes_mask = GDK_WA_X | GDK_WA_Y | GDK_WA_VISUAL | GDK_WA_COLORMAP;
  widget->window = gdk_window_new (widget->parent->window, &attributes, attributes_mask);

  widget->style = gtk_style_attach (widget->style, widget->window);
  gdk_window_set_user_data (widget->window, widget);

  gtk_style_set_background (widget->style, widget->window, GTK_STATE_ACTIVE);
#endif // GTK_VERSION3
}

static Size cvImageWidget_calc_size( int im_width, int im_height, int max_width, int max_height ){
    float aspect = (float)im_width/(float)im_height;
    float max_aspect = (float)max_width/(float)max_height;
    if(aspect > max_aspect){
        return Size( max_width, cvRound(max_width/aspect) );
    }
    return Size( cvRound(max_height*aspect), max_height );
}

#if defined (GTK_VERSION3)
static void
cvImageWidget_get_preferred_width (GtkWidget *widget, gint *minimal_width, gint *natural_width)
{
  g_return_if_fail (widget != NULL);
  g_return_if_fail (CV_IS_IMAGE_WIDGET (widget));
  CvImageWidget * image_widget = CV_IMAGE_WIDGET( widget );

  if(!image_widget->original_image.empty()) {
    *minimal_width = (image_widget->flags & cv::WINDOW_AUTOSIZE) != cv::WINDOW_AUTOSIZE ?
      gdk_window_get_width(gtk_widget_get_window(widget)) : image_widget->original_image.cols;
  }
  else {
    *minimal_width = 320;
  }

  if(!image_widget->scaled_image.empty()) {
    *natural_width = *minimal_width < image_widget->scaled_image.cols ?
      image_widget->scaled_image.cols : *minimal_width;
  }
  else {
    *natural_width = *minimal_width;
  }
}

static void
cvImageWidget_get_preferred_height (GtkWidget *widget, gint *minimal_height, gint *natural_height)
{
  g_return_if_fail (widget != NULL);
  g_return_if_fail (CV_IS_IMAGE_WIDGET (widget));
  CvImageWidget * image_widget = CV_IMAGE_WIDGET( widget );

  if(!image_widget->original_image.empty()) {
    *minimal_height = (image_widget->flags & cv::WINDOW_AUTOSIZE) != cv::WINDOW_AUTOSIZE ?
      gdk_window_get_height(gtk_widget_get_window(widget)) : image_widget->original_image.rows;
  }
  else {
    *minimal_height = 240;
  }

  if(!image_widget->scaled_image.empty()) {
    *natural_height = *minimal_height < image_widget->scaled_image.rows ?
      image_widget->scaled_image.rows : *minimal_height;
  }
  else {
    *natural_height = *minimal_height;
  }
}
#endif //GTK_VERSION3

#if defined (GTK_VERSION2)
static void
cvImageWidget_size_request (GtkWidget      *widget,
                       GtkRequisition *requisition)
{
    CvImageWidget * image_widget = CV_IMAGE_WIDGET( widget );

    //printf("cvImageWidget_size_request ");
    // the case the first time showImageImpl called or when AUTOSIZE
    if( !image_widget->original_image.empty() &&
        ((image_widget->flags & cv::WINDOW_AUTOSIZE) ||
         (image_widget->flags & CV_WINDOW_NO_IMAGE)))
    {
        //printf("original ");
        requisition->width = image_widget->original_image.cols;
        requisition->height = image_widget->original_image.rows;
    }
    // default case
    else if(!image_widget->scaled_image.empty()){
        //printf("scaled ");
        requisition->width = image_widget->scaled_image.cols;
        requisition->height = image_widget->scaled_image.rows;
    }
    // the case before showImageImpl called
    else{
        //printf("default ");
        requisition->width = 320;
        requisition->height = 240;
    }
    //printf("%d %d\n",requisition->width, requisition->height);
}
#endif //GTK_VERSION2

static void cvImageWidget_set_size(GtkWidget * widget, int max_width, int max_height){
    CvImageWidget * image_widget = CV_IMAGE_WIDGET( widget );

    //printf("cvImageWidget_set_size %d %d\n", max_width, max_height);

    // don't allow to set the size
    if(image_widget->flags & cv::WINDOW_AUTOSIZE) return;
    if(image_widget->original_image.empty()) return;

    Size scaled_image_size = cvImageWidget_calc_size( image_widget->original_image.cols,
            image_widget->original_image.rows, max_width, max_height );

    if( !image_widget->scaled_image.empty() &&
            ( image_widget->scaled_image.cols != scaled_image_size.width ||
              image_widget->scaled_image.rows != scaled_image_size.height ))
    {
        image_widget->scaled_image.release();
    }
    if( image_widget->scaled_image.empty() ){
        image_widget->scaled_image.create(scaled_image_size.height, scaled_image_size.width, CV_8UC3);


    }
    CV_Assert(!image_widget->scaled_image.empty());
}

static void
cvImageWidget_size_allocate (GtkWidget     *widget,
                        GtkAllocation *allocation)
{
  CvImageWidget *image_widget;

  //printf("cvImageWidget_size_allocate\n");
  g_return_if_fail (widget != NULL);
  g_return_if_fail (CV_IS_IMAGE_WIDGET (widget));
  g_return_if_fail (allocation != NULL);

#if defined (GTK_VERSION3)
  gtk_widget_set_allocation(widget, allocation);
#elif defined (GTK_VERSION2)
  widget->allocation = *allocation;
#endif //GTK_VERSION3
  image_widget = CV_IMAGE_WIDGET (widget);


  if( (image_widget->flags & cv::WINDOW_AUTOSIZE)==0 && !image_widget->original_image.empty() ){
      // (re) allocated scaled image
      if( image_widget->flags & CV_WINDOW_NO_IMAGE ){
          cvImageWidget_set_size( widget, image_widget->original_image.cols,
                                          image_widget->original_image.rows);
      }
      else{
          cvImageWidget_set_size( widget, allocation->width, allocation->height );
      }
      {
          cv::resize( image_widget->original_image, image_widget->scaled_image, image_widget->scaled_image.size(), 0, 0, cv::INTER_AREA );
      }
  }

  if (gtk_widget_get_realized (widget))
    {
      image_widget = CV_IMAGE_WIDGET (widget);

      if( !image_widget->original_image.empty() &&
              ((image_widget->flags & cv::WINDOW_AUTOSIZE) ||
               (image_widget->flags & CV_WINDOW_NO_IMAGE)) )
      {
#if defined (GTK_VERSION3)
          allocation->width = image_widget->original_image.cols;
          allocation->height = image_widget->original_image.rows;
          gtk_widget_set_allocation(widget, allocation);
#elif defined (GTK_VERSION2)
          widget->allocation.width = image_widget->original_image.cols;
          widget->allocation.height = image_widget->original_image.rows;
#endif //GTK_VERSION3
          gdk_window_move_resize( gtk_widget_get_window(widget),
              allocation->x, allocation->y,
              image_widget->original_image.cols, image_widget->original_image.rows );
          if(image_widget->flags & CV_WINDOW_NO_IMAGE){
              image_widget->flags &= ~CV_WINDOW_NO_IMAGE;
              gtk_widget_queue_resize( GTK_WIDGET(widget) );
          }
      }
      else{
          gdk_window_move_resize (gtk_widget_get_window(widget),
                  allocation->x, allocation->y,
                  allocation->width, allocation->height );
      }
    }
}

#if defined (GTK_VERSION3)
static void
cvImageWidget_destroy (GtkWidget *object)
#else
static void
cvImageWidget_destroy (GtkObject *object)
#endif //GTK_VERSION3
{
  g_return_if_fail (object != NULL);
  g_return_if_fail (CV_IS_IMAGE_WIDGET (object));

#if defined (GTK_VERSION3)
  if (GTK_WIDGET_CLASS (parent_class)->destroy)
    (* GTK_WIDGET_CLASS (parent_class)->destroy) (object);
#else
  if (GTK_OBJECT_CLASS (parent_class)->destroy)
    (* GTK_OBJECT_CLASS (parent_class)->destroy) (object);
#endif //GTK_VERSION3
}

static void cvImageWidget_class_init (gpointer g_class, gpointer /*class_data*/)
{
  CvImageWidgetClass* klass = (CvImageWidgetClass*)g_class;
#if defined (GTK_VERSION3)
  GtkWidgetClass *widget_class = GTK_WIDGET_CLASS (klass);
#endif //GTK_VERSION3
#if defined (GTK_VERSION2)
  GtkObjectClass *object_class = (GtkObjectClass*) klass;
  GtkWidgetClass *widget_class = (GtkWidgetClass*) klass;
#endif //GTK_VERSION2

  parent_class = GTK_WIDGET_CLASS( g_type_class_peek (gtk_widget_get_type ()) );

#if defined (GTK_VERSION3)
  widget_class->destroy = cvImageWidget_destroy;
  widget_class->get_preferred_width = cvImageWidget_get_preferred_width;
  widget_class->get_preferred_height = cvImageWidget_get_preferred_height;
#endif //GTK_VERSION3
#if defined (GTK_VERSION2)
  object_class->destroy = cvImageWidget_destroy;
  widget_class->size_request = cvImageWidget_size_request;
#endif //GTK_VERSION2

  widget_class->realize = cvImageWidget_realize;
  widget_class->size_allocate = cvImageWidget_size_allocate;
  widget_class->button_press_event = NULL;
  widget_class->button_release_event = NULL;
  widget_class->motion_notify_event = NULL;
}

static void
cvImageWidget_init(GTypeInstance* instance, gpointer /*g_class*/)
{
    CvImageWidget* image_widget = (CvImageWidget*)instance;
    image_widget->original_image=0;
    image_widget->scaled_image=0;
    image_widget->flags=0;
}

GType cvImageWidget_get_type (void){
  static GType image_type = 0;

  if (!image_type)
    {
      image_type = g_type_register_static_simple(
          GTK_TYPE_WIDGET,
          (gchar*) "CvImageWidget",
          sizeof(CvImageWidgetClass),
          cvImageWidget_class_init,
          sizeof(CvImageWidget),
          cvImageWidget_init,
          (GTypeFlags)0
          );
    }

  return image_type;
}
/////////////////////////////////////////////////////////////////////////////
// End CvImageWidget
/////////////////////////////////////////////////////////////////////////////


struct CvWindow;

struct CvUIBase {
    CvUIBase(int signature_) : signature(signature_) { }

    int signature;
};

struct CvTrackbar : CvUIBase
{
    CvTrackbar(const std::string& trackbar_name) :
        CvUIBase(CV_TRACKBAR_MAGIC_VAL),
        widget(NULL), name(trackbar_name),
        parent(NULL), data(NULL),
        pos(0), maxval(0), minval(0),
        notify(NULL), notify2(NULL),  // deprecated
        onChangeCallback(NULL), userdata(NULL)
    {
        // nothing
    }
    ~CvTrackbar()
    {
        // destroyed by parent window
    }

    GtkWidget* widget;
    std::string name;
    CvWindow* parent;  // TODO weak_ptr
    int* data;
    int pos;
    int maxval;
    int minval;
    CvTrackbarCallback notify;  // deprecated
    CvTrackbarCallback2 notify2;  // deprecated
    TrackbarCallback onChangeCallback;
    void* userdata;
};


struct CvWindow : CvUIBase
{
    CvWindow(const std::string& window_name) :
        CvUIBase(CV_WINDOW_MAGIC_VAL),
        widget(NULL), frame(NULL), paned(NULL), name(window_name),
        last_key(0), flags(0), status(0),
        on_mouse(NULL), on_mouse_param(NULL)
#ifdef HAVE_OPENGL
        ,useGl(false), glDrawCallback(NULL), glDrawData(NULL), glArea(NULL)
#endif
    {
        CV_LOG_INFO(NULL, "OpenCV/UI: creating GTK window: " << window_name);
    }
    ~CvWindow();
    void destroy();

    GtkWidget* widget;
    GtkWidget* frame;
    GtkWidget* paned;
    std::string name;

    int last_key;
    int flags;
    int status;//0 normal, 1 fullscreen (YV)

    CvMouseCallback on_mouse;
    void* on_mouse_param;

    std::vector< std::shared_ptr<CvTrackbar> > trackbars;

#ifdef HAVE_OPENGL
    bool useGl;

    CvOpenGlDrawCallback glDrawCallback;
    void* glDrawData;
    GtkWidget* glArea;
#endif
};


static gboolean icvOnClose( GtkWidget* widget, GdkEvent* event, gpointer user_data );
static gboolean icvOnKeyPress( GtkWidget* widget, GdkEventKey* event, gpointer user_data );
static void icvOnTrackbar( GtkWidget* widget, gpointer user_data );
static gboolean icvOnMouse( GtkWidget *widget, GdkEvent *event, gpointer user_data );

int thread_started=0;
static gpointer icvWindowThreadLoop(gpointer data);
GMutex*				   last_key_mutex = NULL;
GCond*				   cond_have_key = NULL;
GThread*			   window_thread = NULL;

static int             last_key = -1;

static
std::vector< std::shared_ptr<CvWindow> >& getGTKWindows()
{
    static std::vector< std::shared_ptr<CvWindow> > g_windows;
    return g_windows;
}

static int gtk_InitSystem( int argc, char** argv )
{
    static int wasInitialized = 0;
    static bool hasError = false;

    // check initialization status
    if( !wasInitialized )
    {
        if (!gtk_init_check(&argc, &argv))
        {
            hasError = true;
            wasInitialized = true;
            CV_Error(Error::StsError, "Can't initialize GTK backend");
        }

        setlocale(LC_NUMERIC,"C");

        #if defined(HAVE_OPENGL) && not defined(GTK_VERSION3) // GTK3+ uses GtkGLArea so no need to check for GtkGLExt
            if (!gtk_gl_init_check(&argc, &argv))
            {
                hasError = true;
                wasInitialized = true;
                CV_Error(Error::StsError, "Can't initialize GTK-OpenGL backend");
            }
        #endif

        wasInitialized = 1;
    }
    if (hasError)
       CV_Error(Error::StsError, "GTK backend is not available");

    return 0;
}

int cv::startWindowThread(){
    CV_TRACE_FUNCTION();
    gtk_InitSystem(0,NULL);
    if (!thread_started)
    {
       (void)getWindowMutex();  // force mutex initialization

       // protects the 'last key pressed' variable
       last_key_mutex = g_mutex_new();

       // conditional that indicates a key has been pressed
       cond_have_key = g_cond_new();

       window_thread = g_thread_new("OpenCV window update", icvWindowThreadLoop, NULL);
    }
    thread_started = window_thread!=NULL;
    return thread_started;
}

gpointer icvWindowThreadLoop(gpointer /*data*/)
{
    while(1){
        {
            cv::AutoLock lock(getWindowMutex());
            gtk_main_iteration_do(FALSE);
        }

        // little sleep
        g_usleep(500);

        g_thread_yield();
    }
    return NULL;
}

#define CV_LOCK_MUTEX() cv::AutoLock lock(getWindowMutex())

static
std::shared_ptr<CvWindow> icvFindWindowByName(const std::string& name)
{
    auto& g_windows = getGTKWindows();
    for(size_t i = 0; i < g_windows.size(); ++i)
    {
        auto window = g_windows[i];
        if (!window)
            continue;
        if (window->name == name)
            return window;
    }
    return std::shared_ptr<CvWindow>();
}

static inline
std::shared_ptr<CvWindow> icvFindWindowByName(const char* name)
{
    CV_Assert(name);
    return icvFindWindowByName(std::string(name));
}

static Rect getImageRect_(const std::shared_ptr<CvWindow>& window);

cv::Rect cvGetWindowRect_GTK(const char* name)
{
    CV_Assert(name && "NULL name string");

    CV_LOCK_MUTEX();
    const auto window = icvFindWindowByName(name);
    if (!window)
        CV_Error( cv::Error::StsNullPtr, "NULL window" );

    return getImageRect_(window);
}

#if defined(GTK_VERSION2)
    #define gtk_widget_get_allocated_width(widget) (widget->allocation.width)
    #define gtk_widget_get_allocated_height(widget) (widget->allocation.height)
#endif

static Rect getImageRect_(const std::shared_ptr<CvWindow>& window)
{
    CV_Assert(window);

    gint wx, wy;
#ifdef HAVE_OPENGL
    if (window->useGl) {
        gtk_widget_translate_coordinates(window->widget, gtk_widget_get_toplevel(window->widget), 0, 0, &wx, &wy);
        return Rect(wx, wy, gtk_widget_get_allocated_width(window->widget), gtk_widget_get_allocated_height(window->widget));
    }
#endif

    CvImageWidget * image_widget = CV_IMAGE_WIDGET( window->widget );
    gtk_widget_translate_coordinates(&image_widget->widget, gtk_widget_get_toplevel(&image_widget->widget), 0, 0, &wx, &wy);
    if (!image_widget->scaled_image.empty()) {
      return Rect(wx, wy, MIN(image_widget->scaled_image.cols, gtk_widget_get_allocated_width(window->widget)),
          MIN(image_widget->scaled_image.rows, gtk_widget_get_allocated_height(window->widget)));
    } else if (!image_widget->original_image.empty()) {
      return Rect(wx, wy, MIN(image_widget->original_image.cols, gtk_widget_get_allocated_width(window->widget)),
          MIN(image_widget->original_image.rows, gtk_widget_get_allocated_height(window->widget)));
    }

    return Rect(-1, -1, -1, -1);
}

double cvGetModeWindow_GTK(const char* name)//YV
{
    CV_Assert(name && "NULL name string");

    CV_LOCK_MUTEX();
    const auto window = icvFindWindowByName(name);
    if (!window)
        CV_Error( cv::Error::StsNullPtr, "NULL window" );

    double result = window->status;
    return result;
}

static bool setModeWindow_(const std::shared_ptr<CvWindow>& window, int mode);
void cvSetModeWindow_GTK( const char* name, double prop_value)//Yannick Verdie
{
    CV_Assert(name && "NULL name string");

    CV_LOCK_MUTEX();

    const auto window = icvFindWindowByName(name);
    if (!window)
        CV_Error( cv::Error::StsNullPtr, "NULL window" );

    setModeWindow_(window, (int)prop_value);
}

static bool setModeWindow_(const std::shared_ptr<CvWindow>& window, int mode)
{
    if (window->flags & cv::WINDOW_AUTOSIZE) //if the flag cv::WINDOW_AUTOSIZE is set
        return false;

    //so easy to do fullscreen here, Linux rocks !

    if (window->status == mode)
        return true;

    if (window->status==cv::WINDOW_FULLSCREEN && mode==cv::WINDOW_NORMAL)
    {
        gtk_window_unfullscreen(GTK_WINDOW(window->frame));
        window->status=cv::WINDOW_NORMAL;
        return true;
    }

    if (window->status==cv::WINDOW_NORMAL && mode==cv::WINDOW_FULLSCREEN)
    {
        gtk_window_fullscreen(GTK_WINDOW(window->frame));
        window->status=cv::WINDOW_FULLSCREEN;
        return true;
    }

    return false;
}

void setWindowTitle_GTK(const String& winname, const String& title)
{
    CV_LOCK_MUTEX();

    auto window = icvFindWindowByName(winname.c_str());

    if (!window)
    {
        namedWindow(winname);
        window = icvFindWindowByName(winname.c_str());
        CV_Assert(window);
    }

    gtk_window_set_title(GTK_WINDOW(window->frame), title.c_str());
}

double cvGetPropWindowAutoSize_GTK(const char* name)
{
    CV_Assert(name && "NULL name string");

    CV_LOCK_MUTEX();

    const auto window = icvFindWindowByName(name);
    if (!window)
        return -1; // keep silence here

    double result = window->flags & cv::WINDOW_AUTOSIZE;
    return result;
}

static double getRatioWindow_(const std::shared_ptr<CvWindow>& window);
double cvGetRatioWindow_GTK(const char* name)
{
    CV_Assert(name && "NULL name string");

    CV_LOCK_MUTEX();

    const auto window = icvFindWindowByName(name);
    if (!window)
        return -1; // keep silence here

    return getRatioWindow_(window);
}

static double getRatioWindow_(const std::shared_ptr<CvWindow>& window)
{
    double result = static_cast<double>(
        gtk_widget_get_allocated_width(window->widget)) / gtk_widget_get_allocated_height(window->widget);
    return result;
}

double cvGetOpenGlProp_GTK(const char* name)
{
#ifdef HAVE_OPENGL
    CV_Assert(name && "NULL name string");

    CV_LOCK_MUTEX();

    const auto window = icvFindWindowByName(name);
    if (!window)
        return -1; // keep silence here

    double result = window->useGl;
    return result;
#else
    (void)name;
    return -1;
#endif
}


// OpenGL support

#ifdef HAVE_OPENGL
namespace
{

#ifdef GTK_VERSION3

    void glRealizeCallback(GtkGLArea* area, gpointer user_data) {
        CV_UNUSED(user_data);
        gtk_gl_area_make_current(area);
        if (gtk_gl_area_get_error(area) != NULL)
            CV_Error(cv::Error::OpenGlApiCallError, "OpenGL context is not initialized");
    }

    gboolean glRenderCallback(GtkGLArea* area, GdkGLContext* context, gpointer user_data) {
        CV_UNUSED(context);
        CvWindow* window = (CvWindow*)user_data;
        gtk_gl_area_make_current(area);
        if (gtk_gl_area_get_error(area) != NULL) {
            CV_Error(cv::Error::OpenGlApiCallError, "OpenGL context is not initialized");
            return FALSE;
        }
        if(window->glDrawCallback) {
            window->glDrawCallback(window->glDrawData);
        }
//        gtk_gl_area_queue_render(area);
        return TRUE;
    }

#endif

    void createGlContext(CvWindow* window)
    {
        #ifdef GTK_VERSION3
        g_signal_connect(window->glArea, "realize", G_CALLBACK(glRealizeCallback), window);
        g_signal_connect(window->glArea, "render", G_CALLBACK(glRenderCallback), window);
        #else

        GdkGLConfig* glconfig;

        // Try double-buffered visual
        glconfig = gdk_gl_config_new_by_mode((GdkGLConfigMode)(GDK_GL_MODE_RGB | GDK_GL_MODE_DEPTH | GDK_GL_MODE_DOUBLE));
        if (!glconfig)
            CV_Error( cv::Error::OpenGlApiCallError, "Can't Create A GL Device Context" );

        // Set OpenGL-capability to the widget
        if (!gtk_widget_set_gl_capability(window->widget, glconfig, NULL, TRUE, GDK_GL_RGBA_TYPE))
            CV_Error( cv::Error::OpenGlApiCallError, "Can't Create A GL Device Context" );

        #endif

        window->useGl = true;
    }

    void drawGl(CvWindow* window)
    {
        #ifdef GTK_VERSION3

        GtkGLArea* gtkGlArea = GTK_GL_AREA(window->glArea);
        if (gtk_gl_area_get_error(gtkGlArea) != NULL)
            CV_Error(cv::Error::OpenGlApiCallError, "Can't Activate The GL Rendering Context");

        if (window->glDrawCallback)
            window->glDrawCallback(window->glDrawData);

        #else

        GdkGLContext* glcontext = gtk_widget_get_gl_context(window->widget);
        GdkGLDrawable* gldrawable = gtk_widget_get_gl_drawable(window->widget);

        if (!gdk_gl_drawable_gl_begin (gldrawable, glcontext))
            CV_Error( cv::Error::OpenGlApiCallError, "Can't Activate The GL Rendering Context" );

        glViewport(0, 0, gtk_widget_get_allocated_width(window->widget), gtk_widget_get_allocated_height(window->widget));

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        if (window->glDrawCallback)
            window->glDrawCallback(window->glDrawData);

        if (gdk_gl_drawable_is_double_buffered (gldrawable))
            gdk_gl_drawable_swap_buffers(gldrawable);
        else
            glFlush();

        gdk_gl_drawable_gl_end(gldrawable);

        #endif
    }
}

#endif // HAVE_OPENGL

#if defined (GTK_VERSION2)
static gboolean cvImageWidget_expose(GtkWidget* widget, GdkEventExpose* event, gpointer data)
#elif defined (GTK_VERSION3)
static gboolean cvImageWidget_draw(GtkWidget* widget, cairo_t *cr, gpointer data)
#endif
{
#ifdef HAVE_OPENGL
    CvWindow* window = (CvWindow*)data;

    if (window->useGl)
    {
        drawGl(window);
        return TRUE;
    }
#else
    (void)data;
#endif
#if defined (GTK_VERSION2)
    (void)event;
#endif

  CvImageWidget *image_widget = NULL;
  GdkPixbuf *pixbuf = NULL;

  g_return_val_if_fail (widget != NULL, FALSE);
  g_return_val_if_fail (CV_IS_IMAGE_WIDGET (widget), FALSE);

  image_widget = CV_IMAGE_WIDGET (widget);
#if defined (GTK_VERSION2)
  cairo_t *cr = gdk_cairo_create(widget->window);
#endif

  if( !image_widget->scaled_image.empty() ){
      // center image in available region
      int x0 = (gtk_widget_get_allocated_width(widget) - image_widget->scaled_image.cols)/2;
      int y0 = (gtk_widget_get_allocated_height(widget) - image_widget->scaled_image.rows)/2;

      pixbuf = gdk_pixbuf_new_from_data(image_widget->scaled_image.data, GDK_COLORSPACE_RGB, false,
          8, MIN(image_widget->scaled_image.cols, gtk_widget_get_allocated_width(widget)),
          MIN(image_widget->scaled_image.rows, gtk_widget_get_allocated_height(widget)),
          image_widget->scaled_image.step, NULL, NULL);

      gdk_cairo_set_source_pixbuf(cr, pixbuf, x0, y0);
  }
  else if( !image_widget->original_image.empty() ){
      pixbuf = gdk_pixbuf_new_from_data(image_widget->original_image.data, GDK_COLORSPACE_RGB, false,
          8, MIN(image_widget->original_image.cols, gtk_widget_get_allocated_width(widget)),
          MIN(image_widget->original_image.rows, gtk_widget_get_allocated_height(widget)),
          image_widget->original_image.step, NULL, NULL);
      gdk_cairo_set_source_pixbuf(cr, pixbuf, 0, 0);
  }

  cairo_paint(cr);
  if(pixbuf)
      g_object_unref(pixbuf);
#if defined (GTK_VERSION2)
  cairo_destroy(cr);
#endif
  return TRUE;
}

static std::shared_ptr<CvWindow> namedWindow_(const std::string& name, int flags);

int namedWindowImpl( const char* name, int flags )
{
    gtk_InitSystem(name ? 1 : 0,(char**)&name);
    CV_Assert(name && "NULL name string");

    CV_LOCK_MUTEX();

    // Check the name in the storage
    if (icvFindWindowByName(name))
    {
        return 1;
    }
    auto window = namedWindow_(name, flags);
    return window ? 1 : 0;
}

static std::shared_ptr<CvWindow> namedWindow_(const std::string& name, int flags)
{
    gtk_InitSystem(0, NULL);

    auto window_ptr = std::make_shared<CvWindow>(name);
    CvWindow* window = window_ptr.get();
    window->flags = flags;
    window->status = cv::WINDOW_NORMAL;//YV

    window->frame = gtk_window_new( GTK_WINDOW_TOPLEVEL );

    window->widget = cvImageWidgetNew( flags );

#if defined(HAVE_OPENGL) && defined(GTK_VERSION3)
    if (flags & cv::WINDOW_OPENGL) {
        window->glArea = gtk_gl_area_new();
        gtk_container_add(GTK_CONTAINER(window->frame), window->glArea);
        gtk_widget_show(window->glArea);
    } else {
        window->paned = gtk_vbox_new( FALSE, 0 );
        gtk_box_pack_end( GTK_BOX(window->paned), window->widget, TRUE, TRUE, 0 );
        gtk_widget_show( window->widget );
        gtk_container_add( GTK_CONTAINER(window->frame), window->paned );
        gtk_widget_show( window->paned );
    }
#else
    window->paned = gtk_vbox_new( FALSE, 0 );
    gtk_box_pack_end( GTK_BOX(window->paned), window->widget, TRUE, TRUE, 0 );
    gtk_widget_show( window->widget );
    gtk_container_add( GTK_CONTAINER(window->frame), window->paned );
    gtk_widget_show( window->paned );
#endif

#ifndef HAVE_OPENGL
    if (flags & cv::WINDOW_OPENGL)
        CV_Error( cv::Error::OpenGlNotSupported, "Library was built without OpenGL support" );
#else
    if (flags & cv::WINDOW_OPENGL)
        createGlContext(window);

    window->glDrawCallback = 0;
    window->glDrawData = 0;
#endif

    //
    // configure event handlers
    // TODO -- move this to CvImageWidget ?
    g_signal_connect( window->frame, "key-press-event",
                        G_CALLBACK(icvOnKeyPress), window );
    g_signal_connect( window->widget, "button-press-event",
                        G_CALLBACK(icvOnMouse), window );
    g_signal_connect( window->widget, "button-release-event",
                        G_CALLBACK(icvOnMouse), window );
    g_signal_connect( window->widget, "motion-notify-event",
                        G_CALLBACK(icvOnMouse), window );
    g_signal_connect( window->widget, "scroll-event",
                        G_CALLBACK(icvOnMouse), window );
    g_signal_connect( window->frame, "delete-event",
                        G_CALLBACK(icvOnClose), window );
#if defined(GTK_VERSION3)
    g_signal_connect( window->widget, "draw",
                        G_CALLBACK(cvImageWidget_draw), window );
#else
    g_signal_connect( window->widget, "expose-event",
                        G_CALLBACK(cvImageWidget_expose), window );
#endif //GTK_VERSION3


#if defined(GTK_VERSION3_4)
    gtk_widget_add_events (window->widget, GDK_BUTTON_RELEASE_MASK | GDK_BUTTON_PRESS_MASK | GDK_POINTER_MOTION_MASK | GDK_SCROLL_MASK | GDK_SMOOTH_SCROLL_MASK) ;
#else
    gtk_widget_add_events (window->widget, GDK_BUTTON_RELEASE_MASK | GDK_BUTTON_PRESS_MASK | GDK_POINTER_MOTION_MASK | GDK_SCROLL_MASK) ;
#endif //GTK_VERSION3_4

    gtk_widget_show( window->frame );
    gtk_window_set_title(GTK_WINDOW(window->frame), name.c_str());

    {
        AutoLock lock(getWindowMutex());
        getGTKWindows().push_back(window_ptr);
    }

    bool b_nautosize = ((flags & cv::WINDOW_AUTOSIZE) == 0);
    gtk_window_set_resizable( GTK_WINDOW(window->frame), b_nautosize );

    // allow window to be resized
    if( b_nautosize ){
        GdkGeometry geometry;
        geometry.min_width = 50;
        geometry.min_height = 50;
        gtk_window_set_geometry_hints( GTK_WINDOW( window->frame ), GTK_WIDGET( window->widget ),
            &geometry, (GdkWindowHints) (GDK_HINT_MIN_SIZE));
    }

#ifdef HAVE_OPENGL
    if (window->useGl)
        setOpenGLContextImpl(name.c_str());
#endif

    return window_ptr;
}


#ifdef HAVE_OPENGL

void setOpenGLContextImpl(const char* name)
{
    CV_Assert(name && "NULL name string");

    CV_LOCK_MUTEX();

    auto window = icvFindWindowByName(name);
    if (!window)
        CV_Error( cv::Error::StsNullPtr, "NULL window" );

    if (!window->useGl)
        CV_Error( cv::Error::OpenGlNotSupported, "Window doesn't support OpenGL" );

#ifdef GTK_VERSION3

    if(gtk_gl_area_get_error(GTK_GL_AREA(window->glArea)) != NULL)
        CV_Error( cv::Error::OpenGlApiCallError, "Can't Activate The GL Rendering Context");

#else

    GdkGLContext* glcontext;
    GdkGLDrawable* gldrawable;

    glcontext = gtk_widget_get_gl_context(window->widget);
    gldrawable = gtk_widget_get_gl_drawable(window->widget);

    if (!gdk_gl_drawable_make_current(gldrawable, glcontext))
        CV_Error( cv::Error::OpenGlApiCallError, "Can't Activate The GL Rendering Context" );

#endif

}

void updateWindowImpl(const char* name)
{
    CV_Assert(name && "NULL name string");

    CV_LOCK_MUTEX();

    auto window = icvFindWindowByName(name);
    if (!window)
        return;

    // window does not refresh without this
#ifdef GTK_VERSION3

    if ( GTK_IS_GL_AREA(window->glArea) ){
        gtk_gl_area_queue_render(GTK_GL_AREA(window->glArea));
    } else {
        gtk_widget_queue_draw( GTK_WIDGET(window->widget));
    }

#else

    gtk_widget_queue_draw( GTK_WIDGET(window->widget) );

#endif

}

void setOpenGLDrawCallbackImpl(const char* name, CvOpenGlDrawCallback callback, void* userdata)
{
    CV_Assert(name && "NULL name string");

    CV_LOCK_MUTEX();

    auto window = icvFindWindowByName(name);
    if( !window )
        return;

    if (!window->useGl)
        CV_Error( cv::Error::OpenGlNotSupported, "Window was created without OpenGL context" );

    window->glDrawCallback = callback;
    window->glDrawData = userdata;
}

#endif // HAVE_OPENGL



CvWindow::~CvWindow()
{
    if (frame)
        destroy();
}

inline void CvWindow::destroy()
{
    CV_LOG_INFO(NULL, "OpenCV/UI: destroying GTK window: " << name);
    gtk_widget_destroy(frame);
    frame = nullptr;
}

static void checkLastWindow()
{
    // if last window...
    if (getGTKWindows().empty())
    {
        if( thread_started )
        {
            // send key press signal to jump out of any waiting waitKeyImpl's
            g_cond_broadcast( cond_have_key );
        }
        else
        {
            // Some GTK+ modules (like the Unity module) use GDBusConnection,
            // which has a habit of postponing cleanup by performing it via
            // idle sources added to the main loop. Since this was the last window,
            // we can assume that no event processing is going to happen in the
            // nearest future, so we should force that cleanup (by handling all pending
            // events) while we still have the chance.
            // This is not needed if thread_started is true, because the background
            // thread will process events continuously.
            while( gtk_events_pending() )
                gtk_main_iteration();
        }
    }
}

static
void icvDeleteWindow_( CvWindow* window )
{
    AutoLock lock(getWindowMutex());
    auto& g_windows = getGTKWindows();
    bool found = false;
    for (auto i = g_windows.begin(); i != g_windows.end(); ++i)
    {
        if (i->get() == window)
        {
            g_windows.erase(i);
            found = true;
            break;
        }
    }
    CV_LOG_IF_WARNING(NULL, !found, "OpenCV/GTK: Can't destroy non-registered window");
    checkLastWindow();
}

void destroyWindowImpl( const char* name )
{
    CV_Assert(name && "NULL name string");

    CV_LOCK_MUTEX();
    auto& g_windows = getGTKWindows();

    bool found = false;
    for (auto i = g_windows.begin(); i != g_windows.end(); ++i)
    {
        if (i->get()->name == name)
        {
            g_windows.erase(i);
            found = true;
            break;
        }
    }
    CV_LOG_IF_ERROR(NULL, !found, "OpenCV/GTK: Can't destroy non-registered window: '" << name << "'");

    checkLastWindow();
}


void destroyAllWindowsImpl( void )
{
    CV_LOCK_MUTEX();

    getGTKWindows().clear();
    checkLastWindow();
}

// Size icvCalcOptimalWindowSize( CvWindow * window, Size new_image_size){
//     Size window_size;
//     GtkWidget * toplevel = gtk_widget_get_toplevel( window->frame );
//     gdk_drawable_get_size( GDK_DRAWABLE(toplevel->window),
//             &window_size.width, &window_size.height );

//     window_size.width = window_size.width + new_image_size.width - window->widget->allocation.width;
//     window_size.height = window_size.height + new_image_size.height - window->widget->allocation.height;

//     return window_size;
// }

void showImageImpl( const char* name, InputArray arr )
{
    CV_Assert(name && "NULL name string");

    CV_LOCK_MUTEX();

    auto window = icvFindWindowByName(name);
    if(!window)
    {
        namedWindowImpl(name, 1);
        window = icvFindWindowByName(name);
    }
    CV_Assert(window);

    if (!arr.empty())
    {
    #ifdef HAVE_OPENGL
        if (window->useGl)
        {
            cv::imshow(name, arr);
            return;
        }
    #endif

        CvImageWidget * image_widget = CV_IMAGE_WIDGET( window->widget );
        cvImageWidgetSetImage( image_widget, arr );
    }
}

static void resizeWindow_(const std::shared_ptr<CvWindow>& window, int width, int height);
void resizeWindowImpl(const char* name, int width, int height )
{
    CV_Assert(name && "NULL name string");

    CV_LOCK_MUTEX();

    auto window = icvFindWindowByName(name);
    if(!window)
        return;

    return resizeWindow_(window, width, height);
}

static
void resizeWindow_(const std::shared_ptr<CvWindow>& window, int width, int height)
{
    CV_Assert(window);
    CvImageWidget* image_widget = CV_IMAGE_WIDGET( window->widget );
    //if(image_widget->flags & cv::WINDOW_AUTOSIZE)
        //EXIT;

    gtk_window_set_resizable( GTK_WINDOW(window->frame), 1 );
    gtk_window_resize( GTK_WINDOW(window->frame), width, height );

    // disable initial resize since presumably user wants to keep
    // this window size
    image_widget->flags &= ~CV_WINDOW_NO_IMAGE;
}


void moveWindowImpl( const char* name, int x, int y )
{
    CV_Assert(name && "NULL name string");

    CV_LOCK_MUTEX();

    const auto window = icvFindWindowByName(name);
    if(!window)
        return;

    gtk_window_move( GTK_WINDOW(window->frame), x, y );
}

static
std::shared_ptr<CvTrackbar> icvFindTrackbarByName(const std::shared_ptr<CvWindow>& window, const std::string& name)
{
    CV_Assert(window);
    auto& trackbars = window->trackbars;
    for(size_t i = 0; i < trackbars.size(); ++i)
    {
        auto trackbar = trackbars[i];
        if (!trackbar)
            continue;
        if (trackbar->name == name)
            return trackbar;
    }
    return std::shared_ptr<CvTrackbar>();
}


static int
icvCreateTrackbar( const char* trackbar_name, const char* window_name,
                   int* val, int count, CvTrackbarCallback on_notify,
                   CvTrackbarCallback2 on_notify2, void* userdata )
{
    CV_Assert(window_name && "NULL window name");
    CV_Assert(trackbar_name && "NULL trackbar name");

    if( count <= 0 )
        CV_Error( cv::Error::StsOutOfRange, "Bad trackbar maximal value" );

    CV_LOCK_MUTEX();

    const auto window = icvFindWindowByName(window_name);
    if(!window)
        return 0;

    auto trackbar_ = icvFindTrackbarByName(window, trackbar_name);
    if (!trackbar_)
    {
        trackbar_ = std::make_shared<CvTrackbar>(trackbar_name);
        CvTrackbar* trackbar = trackbar_.get();
        trackbar->parent = window.get();
        window->trackbars.push_back(trackbar_);

        GtkWidget* hscale_box = gtk_hbox_new( FALSE, 10 );
        GtkWidget* hscale_label = gtk_label_new( trackbar_name );
        GtkWidget* hscale = gtk_hscale_new_with_range( 0, count, 1 );
        gtk_scale_set_digits( GTK_SCALE(hscale), 0 );
        //gtk_scale_set_value_pos( hscale, GTK_POS_TOP );
        gtk_scale_set_draw_value( GTK_SCALE(hscale), TRUE );

        trackbar->widget = hscale;
        gtk_box_pack_start( GTK_BOX(hscale_box), hscale_label, FALSE, FALSE, 5 );
        gtk_widget_show( hscale_label );
        gtk_box_pack_start( GTK_BOX(hscale_box), hscale, TRUE, TRUE, 5 );
        gtk_widget_show( hscale );
        gtk_box_pack_start( GTK_BOX(window->paned), hscale_box, FALSE, FALSE, 5 );
        gtk_widget_show( hscale_box );
    }

    CvTrackbar* trackbar = trackbar_.get(); CV_DbgAssert(trackbar);

    if( val )
    {
        int value = *val;
        if( value < 0 )
            value = 0;
        if( value > count )
            value = count;
        gtk_range_set_value( GTK_RANGE(trackbar->widget), value );
        trackbar->pos = value;
        trackbar->data = val;
    }

    trackbar->maxval = count;
    trackbar->notify = on_notify;
    trackbar->notify2 = on_notify2;
    trackbar->userdata = userdata;
    g_signal_connect( trackbar->widget, "value-changed",
                        G_CALLBACK(icvOnTrackbar), trackbar );

    // queue a widget resize to trigger a window resize to
    // compensate for the addition of trackbars
    gtk_widget_queue_resize( GTK_WIDGET(window->widget) );

    return 1;
}

int createTrackbar2Impl( const char* trackbar_name, const char* window_name,
                   int* val, int count, CvTrackbarCallback2 on_notify2,
                   void* userdata )
{
    return icvCreateTrackbar(trackbar_name, window_name, val, count,
                             0, on_notify2, userdata);
}

static
std::shared_ptr<CvTrackbar> createTrackbar_(
    const std::shared_ptr<CvWindow>& window, const std::string& name,
    int count,
    TrackbarCallback onChange, void* userdata
)
{
    CV_Assert(window);
    CV_Assert(!name.empty());

    if (count <= 0)
        CV_Error(Error::StsOutOfRange, "Bad trackbar maximal value");

    auto trackbar_ = std::make_shared<CvTrackbar>(name);
    CvTrackbar* trackbar = trackbar_.get();
    trackbar->parent = window.get();
    window->trackbars.push_back(trackbar_);

    GtkWidget* hscale_box = gtk_hbox_new( FALSE, 10 );
    GtkWidget* hscale_label = gtk_label_new(name.c_str());
    GtkWidget* hscale = gtk_hscale_new_with_range( 0, count, 1 );
    gtk_scale_set_digits( GTK_SCALE(hscale), 0 );
    //gtk_scale_set_value_pos( hscale, GTK_POS_TOP );
    gtk_scale_set_draw_value( GTK_SCALE(hscale), TRUE );

    trackbar->widget = hscale;
    gtk_box_pack_start( GTK_BOX(hscale_box), hscale_label, FALSE, FALSE, 5 );
    gtk_widget_show( hscale_label );
    gtk_box_pack_start( GTK_BOX(hscale_box), hscale, TRUE, TRUE, 5 );
    gtk_widget_show( hscale );
    gtk_box_pack_start( GTK_BOX(window->paned), hscale_box, FALSE, FALSE, 5 );
    gtk_widget_show( hscale_box );

    trackbar->maxval = count;
    trackbar->onChangeCallback = onChange;
    trackbar->userdata = userdata;
    g_signal_connect(trackbar->widget, "value-changed",
                     G_CALLBACK(icvOnTrackbar), trackbar);

    // queue a widget resize to trigger a window resize to
    // compensate for the addition of trackbars
    gtk_widget_queue_resize(GTK_WIDGET(window->widget));

    return trackbar_;
}


void setMouseCallbackImpl( const char* window_name, CvMouseCallback on_mouse, void* param )
{
    CV_Assert(window_name && "NULL window name");

    CV_LOCK_MUTEX();

    const auto window = icvFindWindowByName(window_name);
    if (!window)
        return;

    window->on_mouse = on_mouse;
    window->on_mouse_param = param;
}


int getTrackbarPosImpl( const char* trackbar_name, const char* window_name )
{
    CV_Assert(window_name && "NULL window name");
    CV_Assert(trackbar_name && "NULL trackbar name");

    CV_LOCK_MUTEX();

    const auto window = icvFindWindowByName(window_name);
    if (!window)
        return -1;

    const auto trackbar = icvFindTrackbarByName(window,trackbar_name);
    if (!trackbar)
        return -1;

    return trackbar->pos;
}

static void setTrackbarPos_(const std::shared_ptr<CvTrackbar>& trackbar, int pos);
void setTrackbarPosImpl( const char* trackbar_name, const char* window_name, int pos )
{
    CV_Assert(window_name && "NULL window name");
    CV_Assert(trackbar_name && "NULL trackbar name");

    CV_LOCK_MUTEX();

    const auto window = icvFindWindowByName(window_name);
    if(!window)
        return;

    const auto trackbar = icvFindTrackbarByName(window, trackbar_name);
    if (!trackbar)
    {
        CV_Error( cv::Error::StsNullPtr, "No trackbar found" );
    }

    return setTrackbarPos_(trackbar, pos);
}

static void setTrackbarPos_(const std::shared_ptr<CvTrackbar>& trackbar, int pos)
{
    CV_Assert(trackbar);
    CV_CheckLE(trackbar->minval, trackbar->maxval, "");

    pos = std::max(pos, trackbar->minval);
    pos = std::min(pos, trackbar->maxval);

    gtk_range_set_value( GTK_RANGE(trackbar->widget), pos );
}


void setTrackbarMaxImpl(const char* trackbar_name, const char* window_name, int maxval)
{
    CV_Assert(window_name && "NULL window name");
    CV_Assert(trackbar_name && "NULL trackbar name");

    CV_LOCK_MUTEX();

    const auto window = icvFindWindowByName(window_name);
    if(!window)
        return;

    const auto trackbar = icvFindTrackbarByName(window,trackbar_name);
    if(!trackbar)
        return;

    trackbar->maxval = maxval;
    if (trackbar->maxval >= trackbar->minval)
        gtk_range_set_range(GTK_RANGE(trackbar->widget), trackbar->minval, trackbar->maxval);
}


void setTrackbarMinImpl(const char* trackbar_name, const char* window_name, int minval)
{
    CV_Assert(window_name && "NULL window name");
    CV_Assert(trackbar_name && "NULL trackbar name");

    CV_LOCK_MUTEX();

    const auto window = icvFindWindowByName(window_name);
    if(!window)
        return;

    const auto trackbar = icvFindTrackbarByName(window,trackbar_name);
    if(!trackbar)
        return;

    trackbar->minval = minval;
    if (trackbar->maxval >= trackbar->minval)
        gtk_range_set_range(GTK_RANGE(trackbar->widget), trackbar->minval, trackbar->maxval);
}

static GtkFileFilter* icvMakeGtkFilter(const char* name, const char* patterns, GtkFileFilter* images)
{
    GtkFileFilter* filter = gtk_file_filter_new();
    gtk_file_filter_set_name(filter, name);

    while(patterns[0])
    {
        gtk_file_filter_add_pattern(filter, patterns);
        gtk_file_filter_add_pattern(images, patterns);
        patterns += strlen(patterns) + 1;
    }

    return filter;
}

static void icvShowSaveAsDialog(GtkWidget* widget, CvWindow* window)
{
    if (!window || !widget)
        return;

    CvImageWidget* image_widget = CV_IMAGE_WIDGET(window->widget);
    if (!image_widget || image_widget->original_image.empty())
        return;

    GtkWidget* dialog = gtk_file_chooser_dialog_new("Save As...",
                      GTK_WINDOW(widget),
                      GTK_FILE_CHOOSER_ACTION_SAVE,
                      GTK_STOCK_CANCEL, GTK_RESPONSE_CANCEL,
                      GTK_STOCK_SAVE, GTK_RESPONSE_ACCEPT,
                      NULL);
    gtk_file_chooser_set_do_overwrite_confirmation(GTK_FILE_CHOOSER(dialog), TRUE);

    cv::String sname = gtk_window_get_title(GTK_WINDOW(window->frame));
    sname = sname.substr(sname.find_last_of("\\/") + 1) + ".png";
    gtk_file_chooser_set_current_name(GTK_FILE_CHOOSER(dialog), sname.c_str());

    GtkFileFilter* filter_all = gtk_file_filter_new();
    gtk_file_filter_set_name(filter_all, "All Files");
    gtk_file_filter_add_pattern(filter_all, "*");

    GtkFileFilter* filter_images = gtk_file_filter_new();
    gtk_file_filter_set_name(filter_images, "All Images");

    GtkFileFilter* file_filters[] = {
        icvMakeGtkFilter("Portable Network Graphics files (*.png)",               "*.png\0",                             filter_images),
        icvMakeGtkFilter("JPEG files (*.jpeg;*.jpg;*.jpe)",                       "*.jpeg\0*.jpg\0*.jpe\0",              filter_images),
        icvMakeGtkFilter("Windows bitmap (*.bmp;*.dib)",                          "*.bmp\0*.dib\0",                      filter_images),
        icvMakeGtkFilter("TIFF Files (*.tiff;*.tif)",                             "*.tiff\0*.tif\0",                     filter_images),
        icvMakeGtkFilter("JPEG-2000 files (*.jp2)",                               "*.jp2\0",                             filter_images),
        icvMakeGtkFilter("WebP files (*.webp)",                                   "*.webp\0",                            filter_images),
        icvMakeGtkFilter("Portable image format (*.pbm;*.pgm;*.ppm;*.pxm;*.pnm)", "*.pbm\0*.pgm\0*.ppm\0*.pxm\0*.pnm\0", filter_images),
        icvMakeGtkFilter("OpenEXR Image files (*.exr)",                           "*.exr\0",                             filter_images),
        icvMakeGtkFilter("Radiance HDR (*.hdr;*.pic)",                            "*.hdr\0*.pic\0",                      filter_images),
        icvMakeGtkFilter("Sun raster files (*.sr;*.ras)",                         "*.sr\0*.ras\0",                       filter_images),
        filter_images,
        filter_all
    };

    for (size_t idx = 0; idx < sizeof(file_filters)/sizeof(file_filters[0]); ++idx)
        gtk_file_chooser_add_filter(GTK_FILE_CHOOSER(dialog), file_filters[idx]); // filter ownership is transferred to dialog
    gtk_file_chooser_set_filter(GTK_FILE_CHOOSER(dialog), filter_images);

    cv::String filename;
    if (gtk_dialog_run(GTK_DIALOG(dialog)) == GTK_RESPONSE_ACCEPT)
    {
        char* fname = gtk_file_chooser_get_filename(GTK_FILE_CHOOSER(dialog));
        filename = fname;
        g_free(fname);
    }
    gtk_widget_destroy(dialog);

    if (!filename.empty())
    {
        cv::Mat bgr;
        cv::cvtColor(image_widget->original_image, bgr, cv::COLOR_RGB2BGR);
        cv::imwrite(filename, bgr);
    }
}

#if defined(GTK_VERSION2) && !defined (GDK_KEY_Escape)
#define GDK_KEY_Escape GDK_Escape
#define GDK_KEY_Return GDK_Return
#define GDK_KEY_Linefeed GDK_Linefeed
#define GDK_KEY_Tab GDK_Tab
#define GDK_KEY_s GDK_s
#define GDK_KEY_S GDK_S
#endif //GDK_KEY_Escape

static gboolean icvOnKeyPress(GtkWidget* widget, GdkEventKey* event, gpointer user_data)
{
    int code = 0;

    if ( BIT_ALLIN(event->state, GDK_CONTROL_MASK) && (event->keyval == GDK_KEY_s || event->keyval == GDK_KEY_S))
    {
        try
        {
            icvShowSaveAsDialog(widget, (CvWindow*)user_data);
        }
        catch(...)
        {
            // suppress all exceptions here
        }
    }

    switch( event->keyval )
    {
    case GDK_KEY_Escape:
        code = 27;
        break;
    case GDK_KEY_Return:
    case GDK_KEY_Linefeed:
        code = 13;
        break;
    case GDK_KEY_Tab:
        code = '\t';
    break;
    default:
        code = event->keyval;
    }

    code |= event->state << 16;

    if(thread_started)
    {
        g_mutex_lock(last_key_mutex);
        last_key = code;
        // signal any waiting threads
        g_cond_broadcast(cond_have_key);
        g_mutex_unlock(last_key_mutex);
    }
    else
    {
        last_key = code;
    }

    return FALSE;
}


static void icvOnTrackbar( GtkWidget* widget, gpointer user_data )
{
    int pos = cvRound( gtk_range_get_value(GTK_RANGE(widget)));
    CvTrackbar* trackbar = (CvTrackbar*)user_data;

    if( trackbar && trackbar->signature == CV_TRACKBAR_MAGIC_VAL &&
        trackbar->widget == widget )
    {
        trackbar->pos = pos;
        if (trackbar->onChangeCallback)
            trackbar->onChangeCallback(pos, trackbar->userdata);

        // deprecated
        if( trackbar->data )
            *trackbar->data = pos;
        if( trackbar->notify2 )
            trackbar->notify2(pos, trackbar->userdata);
        else if( trackbar->notify )
            trackbar->notify(pos);
    }
}

static gboolean icvOnClose( GtkWidget* widget, GdkEvent* /*event*/, gpointer user_data )
{
    CvWindow* window = (CvWindow*)user_data;
    if( window->signature == CV_WINDOW_MAGIC_VAL &&
        window->frame == widget )
    {
        try
        {
            icvDeleteWindow_(window);
        }
        catch (...)
        {
            CV_LOG_WARNING(NULL, "OpenCV/GTK: unexpected C++ exception in icvDeleteWindow_");
        }
    }
    return TRUE;
}


static gboolean icvOnMouse( GtkWidget *widget, GdkEvent *event, gpointer user_data )
{
    // TODO move this logic to CvImageWidget
    // TODO add try-catch wrappers into all callbacks
    CvWindow* window = (CvWindow*)user_data;
    if (!window || !widget ||
        window->signature != CV_WINDOW_MAGIC_VAL ||
        window->widget != widget ||
        !window->on_mouse)
        return FALSE;

    Point2f pt32f = {-1., -1.};
    Point pt = {-1,-1};
    int cv_event = -1, state = 0, flags = 0;
    CvImageWidget * image_widget = CV_IMAGE_WIDGET( widget );

    if( event->type == GDK_MOTION_NOTIFY )
    {
        GdkEventMotion* event_motion = (GdkEventMotion*)event;

        cv_event = cv::EVENT_MOUSEMOVE;
        pt32f.x = cvFloor(event_motion->x);
        pt32f.y = cvFloor(event_motion->y);
        state = event_motion->state;
    }
    else if( event->type == GDK_BUTTON_PRESS ||
             event->type == GDK_BUTTON_RELEASE ||
             event->type == GDK_2BUTTON_PRESS )
    {
        GdkEventButton* event_button = (GdkEventButton*)event;
        pt32f.x = cvFloor(event_button->x);
        pt32f.y = cvFloor(event_button->y);


        if( event_button->type == GDK_BUTTON_PRESS )
        {
            cv_event = event_button->button == 1 ? cv::EVENT_LBUTTONDOWN :
                       event_button->button == 2 ? cv::EVENT_MBUTTONDOWN :
                       event_button->button == 3 ? cv::EVENT_RBUTTONDOWN : 0;
        }
        else if( event_button->type == GDK_BUTTON_RELEASE )
        {
            cv_event = event_button->button == 1 ? cv::EVENT_LBUTTONUP :
                       event_button->button == 2 ? cv::EVENT_MBUTTONUP :
                       event_button->button == 3 ? cv::EVENT_RBUTTONUP : 0;
        }
        else if( event_button->type == GDK_2BUTTON_PRESS )
        {
            cv_event = event_button->button == 1 ? cv::EVENT_LBUTTONDBLCLK :
                       event_button->button == 2 ? cv::EVENT_MBUTTONDBLCLK :
                       event_button->button == 3 ? cv::EVENT_RBUTTONDBLCLK : 0;
        }
        state = event_button->state;
    }
    else if( event->type == GDK_SCROLL )
    {
        GdkEventScroll* event_scroll = (GdkEventScroll*)event;
        pt32f.x = cvFloor(event_scroll->x);
        pt32f.y = cvFloor(event_scroll->y);

#if defined(GTK_VERSION3_4)
        // NOTE: in current implementation doesn't possible to put into callback function delta_x and delta_y separately
        double delta = (event->scroll.delta_x + event->scroll.delta_y);
        cv_event   = (event->scroll.delta_x==0) ? cv::EVENT_MOUSEWHEEL : cv::EVENT_MOUSEHWHEEL;
#else
        cv_event = cv::EVENT_MOUSEWHEEL;
#endif //GTK_VERSION3_4

        state    = event->scroll.state;

        switch(event->scroll.direction) {
#if defined(GTK_VERSION3_4)
        case GDK_SCROLL_SMOOTH: flags |= (((int)delta << 16));
            break;
#endif //GTK_VERSION3_4
        case GDK_SCROLL_LEFT:  cv_event = cv::EVENT_MOUSEHWHEEL;
            /* FALLTHRU */
        case GDK_SCROLL_UP:    flags |= ~0xffff;
            break;
        case GDK_SCROLL_RIGHT: cv_event = cv::EVENT_MOUSEHWHEEL;
            /* FALLTHRU */
        case GDK_SCROLL_DOWN:  flags |= (((int)1 << 16));
            break;
        default: ;
        };
    }

    if( cv_event >= 0 )
    {
        // scale point if image is scaled
        if( (image_widget->flags & cv::WINDOW_AUTOSIZE)==0 &&
             !image_widget->original_image.empty() &&
             !image_widget->scaled_image.empty() )
        {
            // image origin is not necessarily at (0,0)
            int x0 = (gtk_widget_get_allocated_width(widget) - image_widget->scaled_image.cols)/2;
            int y0 = (gtk_widget_get_allocated_height(widget) - image_widget->scaled_image.rows)/2;
            pt.x = cvFloor( ((pt32f.x-x0)*image_widget->original_image.cols)/
                                            image_widget->scaled_image.cols );
            pt.y = cvFloor( ((pt32f.y-y0)*image_widget->original_image.rows)/
                                            image_widget->scaled_image.rows );
        }
        else
        {
            pt = pt32f;
        }

        if (image_widget->original_image.empty()/*OpenGL*/ || (
               (unsigned)pt.x < (unsigned)(image_widget->original_image.size().width) &&
               (unsigned)pt.y < (unsigned)(image_widget->original_image.size().height)
            ))
        {
            // handle non-keyboard (mouse) modifiers first
            flags |=
                BIT_MAP(state, GDK_BUTTON1_MASK, cv::EVENT_FLAG_LBUTTON)  |
                BIT_MAP(state, GDK_BUTTON2_MASK, cv::EVENT_FLAG_MBUTTON)  |
                BIT_MAP(state, GDK_BUTTON3_MASK, cv::EVENT_FLAG_RBUTTON);
            // keyboard modifiers
            state &= gtk_accelerator_get_default_mod_mask();
            flags |=
                BIT_MAP(state, GDK_SHIFT_MASK,   cv::EVENT_FLAG_SHIFTKEY) |
                BIT_MAP(state, GDK_CONTROL_MASK, cv::EVENT_FLAG_CTRLKEY)  |
                BIT_MAP(state, GDK_MOD1_MASK,    cv::EVENT_FLAG_ALTKEY)   |
                BIT_MAP(state, GDK_MOD2_MASK,    cv::EVENT_FLAG_ALTKEY);
            window->on_mouse( cv_event, pt.x, pt.y, flags, window->on_mouse_param );
        }
    }

    return FALSE;
}


static gboolean icvAlarm( gpointer user_data )
{
    *(int*)user_data = 1;
    return FALSE;
}


int waitKeyImpl( int delay )
{
    if (thread_started && g_thread_self() != window_thread)
    {
        gboolean expired = true;
        int my_last_key;

        g_mutex_lock(last_key_mutex);
        // wait for signal or timeout if delay > 0
        if(delay>0){
            GTimeVal timer;
            g_get_current_time(&timer);
            g_time_val_add(&timer, delay*1000);
            expired = !g_cond_timed_wait(cond_have_key, last_key_mutex, &timer);
        }
        else{
            if (getGTKWindows().empty())
            {
                CV_LOG_WARNING(NULL, "cv::waitKey() is called without timeout and missing active windows. Ignoring");
            }
            else
            {
                g_cond_wait(cond_have_key, last_key_mutex);
                expired=false;
            }
        }
        my_last_key = last_key;
        g_mutex_unlock(last_key_mutex);
        if (expired || getGTKWindows().empty())
        {
            return -1;
        }
        return my_last_key;
    }
    else
    {
        int expired = 0;
        guint timer = 0;
        if( delay > 0 )
            timer = g_timeout_add( delay, icvAlarm, &expired );
        last_key = -1;
        while( gtk_main_iteration_do(TRUE) && last_key < 0 && !expired && (delay > 0 || !getGTKWindows().empty()))
            ;

        if( delay > 0 && !expired )
            g_source_remove(timer);
    }
    return last_key;
}

namespace cv { namespace impl {

using namespace cv::highgui_backend;

class GTKTrackbar;

class GTKWindow
        : public UIWindow
        , public std::enable_shared_from_this<GTKWindow>
{
protected:
    const std::string name_;
    std::weak_ptr<CvWindow> window_;
    std::map<std::string, std::shared_ptr<GTKTrackbar> > trackbars_;
public:
    GTKWindow(const std::string& name, const std::shared_ptr<CvWindow>& window)
        : name_(name)
        , window_(window)
    {
        // nothing
    }

    ~GTKWindow() CV_OVERRIDE
    {
        if (!window_.expired())
            destroy();
        CV_LOG_DEBUG(NULL, "OpenCV/UI/GTK: GTKWindow(" << name_ << ") is disposed");
    }

    const std::string& getID() const CV_OVERRIDE { return name_; }

    bool isActive() const CV_OVERRIDE { return !window_.expired(); }

    void destroy() CV_OVERRIDE
    {
        cv::AutoLock lock(getWindowMutex());
        if (!window_.expired())
        {
            auto window = window_.lock();
            if (window)
                window->destroy();
            window_.reset();
        }
    }

    void imshow(InputArray image) CV_OVERRIDE
    {
        auto window = window_.lock();
        CV_Assert(window);
        CvImageWidget* image_widget = CV_IMAGE_WIDGET(window->widget);
        CV_Assert(image_widget);
        cvImageWidgetSetImage(image_widget, image);
    }

    double getProperty(int prop) const CV_OVERRIDE
    {
        auto window = window_.lock();
        CV_Assert(window);
        switch (prop)
        {
        case cv::WND_PROP_FULLSCREEN:
            return (double)window->status;

        case cv::WND_PROP_AUTOSIZE:
            return (window->flags & cv::WINDOW_AUTOSIZE) ? 1.0 : 0.0;

        case cv::WND_PROP_ASPECT_RATIO:
            return getRatioWindow_(window);

#ifdef HAVE_OPENGL
        case cv::WND_PROP_OPENGL:
            return window->useGl ? 1.0 : 0.0;
#endif

        default:
            break;
        }
        return std::numeric_limits<double>::quiet_NaN();
    }

    bool setProperty(int prop, double value) CV_OVERRIDE
    {
        auto window = window_.lock();
        CV_Assert(window);
        switch (prop)
        {
        case cv::WND_PROP_FULLSCREEN:
            if ((int)value != cv::WINDOW_NORMAL && (int)value != cv::WINDOW_FULLSCREEN)  // bad arg
                break;
            setModeWindow_(window, value);
            return true;

        default:
            break;
        }
        return false;
    }

    void resize(int width, int height) CV_OVERRIDE
    {
        auto window = window_.lock();
        CV_Assert(window);
        resizeWindow_(window, width, height);
    }

    void move(int x, int y) CV_OVERRIDE
    {
        auto window = window_.lock();
        CV_Assert(window);
        gtk_window_move(GTK_WINDOW(window->frame), x, y);
    }

    Rect getImageRect() const CV_OVERRIDE
    {
        auto window = window_.lock();
        CV_Assert(window);
        return getImageRect_(window);
    }

    void setTitle(const std::string& title) CV_OVERRIDE
    {
        auto window = window_.lock();
        CV_Assert(window);
        gtk_window_set_title(GTK_WINDOW(window->frame), title.c_str());
    }

    void setMouseCallback(MouseCallback onMouse, void* userdata /*= 0*/) CV_OVERRIDE
    {
        auto window = window_.lock();
        CV_Assert(window);
        window->on_mouse = onMouse;
        window->on_mouse_param = userdata;
    }

    std::shared_ptr<UITrackbar> createTrackbar(
        const std::string& name,
        int count,
        TrackbarCallback onChange /*= 0*/,
        void* userdata /*= 0*/
    ) CV_OVERRIDE
    {
        auto window = window_.lock();
        CV_Assert(window);
        CV_LOG_INFO(NULL, "OpenCV/UI: Creating GTK trackbar at '" << name_ << "': '" << name << "'");
        auto trackbar = createTrackbar_(window, name, count, onChange, userdata);
        auto ui_trackbar = std::make_shared<GTKTrackbar>(name, trackbar, shared_from_this());
        {
            cv::AutoLock lock(getWindowMutex());
            trackbars_.emplace(name, ui_trackbar);
        }
        return std::static_pointer_cast<UITrackbar>(ui_trackbar);
    }

    std::shared_ptr<UITrackbar> findTrackbar(const std::string& name) CV_OVERRIDE
    {
        cv::AutoLock lock(getWindowMutex());
        auto i = trackbars_.find(name);
        if (i != trackbars_.end())
        {
            return std::static_pointer_cast<UITrackbar>(i->second);
        }
        return std::shared_ptr<UITrackbar>();
    }
};  // GTKWindow


class GTKTrackbar : public UITrackbar
{
protected:
    /*const*/ std::string name_;
    std::weak_ptr<CvTrackbar> trackbar_;
    std::weak_ptr<GTKWindow> parent_;
    std::map<std::string, std::shared_ptr<GTKTrackbar> > trackbars_;
public:
    GTKTrackbar(const std::string& name, const std::shared_ptr<CvTrackbar>& trackbar, const std::shared_ptr<GTKWindow>& parent)
        : trackbar_(trackbar)
        , parent_(parent)
    {
        name_ = std::string("<") + name + ">@" + parent->getID();
    }

    ~GTKTrackbar() CV_OVERRIDE
    {
        if (!trackbar_.expired())
            destroy();
        CV_LOG_DEBUG(NULL, "OpenCV/UI/GTK: GTKTrackbar(" << name_ << ") is disposed");
    }

    const std::string& getID() const CV_OVERRIDE { return name_; }

    bool isActive() const CV_OVERRIDE { return !trackbar_.expired(); }

    void destroy() CV_OVERRIDE
    {
        // nothing (destroyed with parent window, dedicated trackbar removal is not supported)
    }

    int getPos() const CV_OVERRIDE
    {
        auto trackbar = trackbar_.lock();
        CV_Assert(trackbar);
        return trackbar->pos;
    }
    void setPos(int pos) CV_OVERRIDE
    {
        auto trackbar = trackbar_.lock();
        CV_Assert(trackbar);
        return setTrackbarPos_(trackbar, pos);
    }

    cv::Range getRange() const CV_OVERRIDE
    {
        auto trackbar = trackbar_.lock();
        CV_Assert(trackbar);
        return cv::Range(trackbar->minval, trackbar->maxval);
    }

    void setRange(const cv::Range& range) CV_OVERRIDE
    {
        auto trackbar = trackbar_.lock();
        CV_Assert(trackbar);
        CV_CheckLE(range.start, range.end, "Invalid trackbar range");
        gtk_range_set_range(GTK_RANGE(trackbar->widget), range.start, range.end);
    }
};  // GTKTrackbar


class GTKBackendUI : public UIBackend
{
public:
    GTKBackendUI()
    {
        // NB: avoid static initialization order fiasco
        (void)getGTKWindows();
    }
    ~GTKBackendUI() CV_OVERRIDE
    {
        destroyAllWindows();
    }

    void destroyAllWindows() CV_OVERRIDE
    {
        destroyAllWindowsImpl();
    }

    // namedWindow
    virtual std::shared_ptr<UIWindow> createWindow(
        const std::string& winname,
        int flags
    ) CV_OVERRIDE
    {
        CV_LOG_INFO(NULL, "OpenCV/UI: Creating GTK window: " << winname << " (" << flags << ")");
        auto window = namedWindow_(winname, flags);
        auto ui_window = std::make_shared<GTKWindow>(winname, window);
        return ui_window;
    }

    int waitKeyEx(int delay) CV_OVERRIDE
    {
        return waitKeyImpl(delay);
    }
    int pollKey() CV_OVERRIDE
    {
        return waitKeyImpl(1);  // TODO
    }

    const std::string getName() const CV_OVERRIDE
    {
#if GTK_MAJOR_VERSION == 2
        return "GTK2";
#elif GTK_MAJOR_VERSION == 3
        return "GTK3";
#elif GTK_MAJOR_VERSION == 4
        return "GTK4";
#else
#error "Unsupported GTK version"
#endif
    }
};  // GTKBackendUI

static
std::shared_ptr<GTKBackendUI>& getInstance()
{
    static std::shared_ptr<GTKBackendUI> g_instance = std::make_shared<GTKBackendUI>();
    return g_instance;
}

} // namespace impl

#ifndef BUILD_PLUGIN
namespace highgui_backend {

std::shared_ptr<UIBackend> createUIBackendGTK()
{
    return impl::getInstance();
}

}  // namespace highgui_backend
#endif

}  // namespace

#ifdef BUILD_PLUGIN

#define ABI_VERSION 0
#define API_VERSION 0
#include "plugin_api.hpp"

static
CvResult cv_getInstance(CV_OUT CvPluginUIBackend* handle) CV_NOEXCEPT
{
    try
    {
        if (!handle)
            return CV_ERROR_FAIL;
        *handle = cv::impl::getInstance().get();
        return CV_ERROR_OK;
    }
    catch (...)
    {
        return CV_ERROR_FAIL;
    }
}

static const OpenCV_UI_Plugin_API plugin_api =
{
    {
        sizeof(OpenCV_UI_Plugin_API), ABI_VERSION, API_VERSION,
        CV_VERSION_MAJOR, CV_VERSION_MINOR, CV_VERSION_REVISION, CV_VERSION_STATUS,
        "GTK" CVAUX_STR(GTK_MAJOR_VERSION) " OpenCV UI plugin"
    },
    {
        /*  1*/cv_getInstance
    }
};

const OpenCV_UI_Plugin_API* CV_API_CALL opencv_ui_plugin_init_v0(int requested_abi_version, int requested_api_version, void* /*reserved=NULL*/) CV_NOEXCEPT
{
    if (requested_abi_version == ABI_VERSION && requested_api_version <= API_VERSION)
        return &plugin_api;
    return NULL;
}

#endif  // BUILD_PLUGIN

#endif  // HAVE_GTK
