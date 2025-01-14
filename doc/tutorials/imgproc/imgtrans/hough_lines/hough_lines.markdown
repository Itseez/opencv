Hough Line Transform {#tutorial_hough_lines}
====================

@tableofcontents

@prev_tutorial{tutorial_canny_detector}
@next_tutorial{tutorial_hough_circle}

|    |    |
| -: | :- |
| Original author | Ana Huamán |
| Compatibility | OpenCV >= 3.0 |

Goal
----

In this tutorial you will learn how to:

-   Use the OpenCV functions **HoughLines()** and **HoughLinesP()** to detect lines in an
    image.

Theory
------

@note The explanation below belongs to the book **Learning OpenCV** by Bradski and Kaehler.

Hough Line Transform
--------------------

-# The Hough Line Transform is a transform used to detect straight lines.
-# To apply the Transform, first an edge detection pre-processing is desirable.

### How does it work?

-#  As you know, a line in the image space can be expressed with two variables. For example:

    -#  In the **Cartesian coordinate system:** Parameters: \f$(m,b)\f$.
    -#  In the **Polar coordinate system:** Parameters: \f$(r,\theta)\f$

    ![](images/Hough_Lines_Tutorial_Theory_0.jpg)

    For Hough Transforms, we will express lines in the *Polar system*. Hence, a line equation can be
    written as:

    \f[y = \left ( -\dfrac{\cos \theta}{\sin \theta} \right ) x + \left ( \dfrac{r}{\sin \theta} \right )\f]

Arranging the terms: \f$r = x \cos \theta + y \sin \theta\f$

-#  In general for each point \f$(x_{0}, y_{0})\f$, we can define the family of lines that goes through
    that point as:

    \f[r_{\theta} = x_{0} \cdot \cos \theta  + y_{0} \cdot \sin \theta\f]

    Meaning that each pair \f$(r_{\theta},\theta)\f$ represents each line that passes by
    \f$(x_{0}, y_{0})\f$.

-#  If for a given \f$(x_{0}, y_{0})\f$ we plot the family of lines that goes through it, we get a
    sinusoid. For instance, for \f$x_{0} = 8\f$ and \f$y_{0} = 6\f$ we get the following plot (in a plane
    \f$\theta\f$ - \f$r\f$):

    ![](images/Hough_Lines_Tutorial_Theory_1.jpg)

    We consider only points such that \f$r > 0\f$ and \f$0< \theta < 2 \pi\f$.

-#  We can do the same operation above for all the points in an image. If the curves of two
    different points intersect in the plane \f$\theta\f$ - \f$r\f$, that means that both points belong to a
    same line. For instance, following with the example above and drawing the plot for two more
    points: \f$x_{1} = 4\f$, \f$y_{1} = 9\f$ and \f$x_{2} = 12\f$, \f$y_{2} = 3\f$, we get:

    ![](images/Hough_Lines_Tutorial_Theory_2.jpg)

    The three plots intersect in one single point \f$(0.925, 9.6)\f$, these coordinates are the
    parameters (\f$\theta, r\f$) or the line in which \f$(x_{0}, y_{0})\f$, \f$(x_{1}, y_{1})\f$ and
    \f$(x_{2}, y_{2})\f$ lay.

-#  What does all the stuff above mean? It means that in general, a line can be *detected* by
    finding the number of intersections between curves.The more curves intersecting means that the
    line represented by that intersection have more points. In general, we can define a *threshold*
    of the minimum number of intersections needed to *detect* a line.
-#  This is what the Hough Line Transform does. It keeps track of the intersection between curves of
    every point in the image. If the number of intersections is above some *threshold*, then it
    declares it as a line with the parameters \f$(\theta, r_{\theta})\f$ of the intersection point.

### Standard and Probabilistic Hough Line Transform

OpenCV implements three kind of Hough Line Transforms:

a.  **The Standard Hough Transform**

-   It consists in pretty much what we just explained in the previous section. It gives you as
    result a vector of couples \f$(\theta, r_{\theta})\f$
-   In OpenCV it is implemented with the function **HoughLines()**

b.  **The Probabilistic Hough Line Transform**

-   A more efficient implementation of the Hough Line Transform. It gives as output the extremes
    of the detected lines \f$(x_{0}, y_{0}, x_{1}, y_{1})\f$
-   In OpenCV it is implemented with the function **HoughLinesP()**

c.  **The Weighted Hough Transform**

-   Uses edge intensity instead binary 0 or 1 values in standard Hough transform.
-   In OpenCV it is implemented with the function **HoughLines()** with use_edgeval=true.
-   See the example in samples/cpp/tutorial_code/ImgTrans/HoughLines_Demo.cpp.

###  What does this program do?
    -   Loads an image
    -   Applies a *Standard Hough Line Transform* and a *Probabilistic Line Transform*.
    -   Display the original image and the detected line in three windows.

Code
----

@add_toggle_cpp
The sample code that we will explain can be downloaded from
[here](https://raw.githubusercontent.com/opencv/opencv/5.x/samples/cpp/tutorial_code/ImgTrans/houghlines.cpp).
A slightly fancier version (which shows both Hough standard and probabilistic
with trackbars for changing the threshold values) can be found
[here](https://raw.githubusercontent.com/opencv/opencv/5.x/samples/cpp/tutorial_code/ImgTrans/HoughLines_Demo.cpp).
@include samples/cpp/tutorial_code/ImgTrans/houghlines.cpp
@end_toggle

@add_toggle_java
The sample code that we will explain can be downloaded from
[here](https://raw.githubusercontent.com/opencv/opencv/5.x/samples/java/tutorial_code/ImgTrans/HoughLine/HoughLines.java).
@include samples/java/tutorial_code/ImgTrans/HoughLine/HoughLines.java
@end_toggle

@add_toggle_python
The sample code that we will explain can be downloaded from
[here](https://raw.githubusercontent.com/opencv/opencv/5.x/samples/python/tutorial_code/ImgTrans/HoughLine/hough_lines.py).
@include samples/python/tutorial_code/ImgTrans/HoughLine/hough_lines.py
@end_toggle

Explanation
-----------

### Load an image:

@add_toggle_cpp
@snippet samples/cpp/tutorial_code/ImgTrans/houghlines.cpp load
@end_toggle

@add_toggle_java
@snippet samples/java/tutorial_code/ImgTrans/HoughLine/HoughLines.java load
@end_toggle

@add_toggle_python
@snippet samples/python/tutorial_code/ImgTrans/HoughLine/hough_lines.py load
@end_toggle

### Detect the edges of the image by using a Canny detector:

@add_toggle_cpp
@snippet samples/cpp/tutorial_code/ImgTrans/houghlines.cpp edge_detection
@end_toggle

@add_toggle_java
@snippet samples/java/tutorial_code/ImgTrans/HoughLine/HoughLines.java edge_detection
@end_toggle

@add_toggle_python
@snippet samples/python/tutorial_code/ImgTrans/HoughLine/hough_lines.py edge_detection
@end_toggle

Now we will apply the Hough Line Transform. We will explain how to use both OpenCV functions
available for this purpose.

### Standard Hough Line Transform:
First, you apply the Transform:

@add_toggle_cpp
@snippet samples/cpp/tutorial_code/ImgTrans/houghlines.cpp hough_lines
@end_toggle

@add_toggle_java
@snippet samples/java/tutorial_code/ImgTrans/HoughLine/HoughLines.java hough_lines
@end_toggle

@add_toggle_python
@snippet samples/python/tutorial_code/ImgTrans/HoughLine/hough_lines.py hough_lines
@end_toggle

-       with the following arguments:

        -   *dst*: Output of the edge detector. It should be a grayscale image (although in fact it
            is a binary one)
        -   *lines*: A vector that will store the parameters \f$(r,\theta)\f$ of the detected lines
        -   *rho* : The resolution of the parameter \f$r\f$ in pixels. We use **1** pixel.
        -   *theta*: The resolution of the parameter \f$\theta\f$ in radians. We use **1 degree**
            (CV_PI/180)
        -   *threshold*: The minimum number of intersections to "*detect*" a line
        -   *srn* and *stn*: Default parameters to zero. Check OpenCV reference for more info.

And then you display the result by drawing the lines.
@add_toggle_cpp
@snippet samples/cpp/tutorial_code/ImgTrans/houghlines.cpp draw_lines
@end_toggle

@add_toggle_java
@snippet samples/java/tutorial_code/ImgTrans/HoughLine/HoughLines.java draw_lines
@end_toggle

@add_toggle_python
@snippet samples/python/tutorial_code/ImgTrans/HoughLine/hough_lines.py draw_lines
@end_toggle

### Probabilistic Hough Line Transform
First you apply the transform:

@add_toggle_cpp
@snippet samples/cpp/tutorial_code/ImgTrans/houghlines.cpp hough_lines_p
@end_toggle

@add_toggle_java
@snippet samples/java/tutorial_code/ImgTrans/HoughLine/HoughLines.java hough_lines_p
@end_toggle

@add_toggle_python
@snippet samples/python/tutorial_code/ImgTrans/HoughLine/hough_lines.py hough_lines_p
@end_toggle

-       with the arguments:

        -   *dst*: Output of the edge detector. It should be a grayscale image (although in fact it
            is a binary one)
        -   *lines*: A vector that will store the parameters
            \f$(x_{start}, y_{start}, x_{end}, y_{end})\f$ of the detected lines
        -   *rho* : The resolution of the parameter \f$r\f$ in pixels. We use **1** pixel.
        -   *theta*: The resolution of the parameter \f$\theta\f$ in radians. We use **1 degree**
            (CV_PI/180)
        -   *threshold*: The minimum number of intersections to "*detect*" a line
        -   *minLineLength*: The minimum number of points that can form a line. Lines with less than
            this number of points are disregarded.
        -   *maxLineGap*: The maximum gap between two points to be considered in the same line.

And then you display the result by drawing the lines.

@add_toggle_cpp
@snippet samples/cpp/tutorial_code/ImgTrans/houghlines.cpp draw_lines_p
@end_toggle

@add_toggle_java
@snippet samples/java/tutorial_code/ImgTrans/HoughLine/HoughLines.java draw_lines_p
@end_toggle

@add_toggle_python
@snippet samples/python/tutorial_code/ImgTrans/HoughLine/hough_lines.py draw_lines_p
@end_toggle

### Display the original image and the detected lines:

@add_toggle_cpp
@snippet samples/cpp/tutorial_code/ImgTrans/houghlines.cpp imshow
@end_toggle

@add_toggle_java
@snippet samples/java/tutorial_code/ImgTrans/HoughLine/HoughLines.java imshow
@end_toggle

@add_toggle_python
@snippet samples/python/tutorial_code/ImgTrans/HoughLine/hough_lines.py imshow
@end_toggle

### Wait until the user exits the program

@add_toggle_cpp
@snippet samples/cpp/tutorial_code/ImgTrans/houghlines.cpp exit
@end_toggle

@add_toggle_java
@snippet samples/java/tutorial_code/ImgTrans/HoughLine/HoughLines.java exit
@end_toggle

@add_toggle_python
@snippet samples/python/tutorial_code/ImgTrans/HoughLine/hough_lines.py exit
@end_toggle

Result
------

@note
   The results below are obtained using the slightly fancier version we mentioned in the *Code*
    section. It still implements the same stuff as above, only adding the Trackbar for the
    Threshold.

Using an input image such as a [sudoku image](https://raw.githubusercontent.com/opencv/opencv/5.x/samples/data/sudoku.png).
We get the following result by using the Standard Hough Line Transform:
![](images/hough_lines_result1.png)
And by using the Probabilistic Hough Line Transform:
![](images/hough_lines_result2.png)

You may observe that the number of lines detected vary while you change the *threshold*. The
explanation is sort of evident: If you establish a higher threshold, fewer lines will be detected
(since you will need more points to declare a line detected).
