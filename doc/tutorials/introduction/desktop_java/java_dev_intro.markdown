Introduction to Java Development {#tutorial_java_dev_intro}
================================

@prev_tutorial{tutorial_windows_visual_studio_image_watch}
@next_tutorial{tutorial_java_eclipse}

|    |    |
| -: | :- |
| Original author | Eric Christiansen and Andrey Pavlenko |
| Compatibility | OpenCV >= 3.0 |

@tableofcontents

@warning
This tutorial can contain obsolete information.

As of OpenCV 2.4.4, OpenCV supports desktop Java development using nearly the same interface as for
Android development. This guide will help you to create your first Java (or Scala) application using
OpenCV. We will use either [Apache Ant](http://ant.apache.org/) or [Simple Build Tool
(SBT)](http://www.scala-sbt.org/) to build the application.

If you want to use Eclipse head to @ref tutorial_java_eclipse. For further reading after this guide, look at
the @ref tutorial_android_dev_intro tutorials.

What we'll do in this guide
---------------------------

In this guide, we will:

-   Get OpenCV with desktop Java support
-   Create an Ant or SBT project
-   Write a simple OpenCV application in Java or Scala

The same process was used to create the samples in the `samples/java` folder of the OpenCV
repository, so consult those files if you get lost.

Get proper OpenCV
-----------------

Starting from version 2.4.4 OpenCV includes desktop Java bindings.

### Download

The most simple way to get it is downloading the appropriate package of **version 2.4.4 or higher**
from the [OpenCV SourceForge repository](http://sourceforge.net/projects/opencvlibrary/files/).

@note Windows users can find the prebuilt files needed for Java development in the
`opencv/build/java/` folder inside the package. For other OSes it's required to build OpenCV from
sources.

Another option to get OpenCV sources is to clone [OpenCV git
repository](https://github.com/opencv/opencv/). In order to build OpenCV with Java bindings you need
JDK (Java Development Kit) (we recommend [Oracle/Sun JDK 6 or
7](http://www.oracle.com/technetwork/java/javase/downloads/)), [Apache Ant](http://ant.apache.org/)
and Python v2.6 or higher to be installed.

### Build

Let's build OpenCV:
@code{.bash}
git clone git://github.com/opencv/opencv.git
cd opencv
git checkout 2.4
mkdir build
cd build
@endcode
Generate a Makefile or a MS Visual Studio\* solution, or whatever you use for building executables
in your system:
@code{.bash}
cmake -DBUILD_SHARED_LIBS=OFF ..
@endcode
or
@code{.bat}
cmake -DBUILD_SHARED_LIBS=OFF -G "Visual Studio 10" ..
@endcode

@note When OpenCV is built as a set of **static** libraries (-DBUILD_SHARED_LIBS=OFF option) the
Java bindings dynamic library is all-sufficient, i.e. doesn't depend on other OpenCV libs, but
includes all the OpenCV code inside.

Examine the output of CMake and ensure java is one of the
modules "To be built". If not, it's likely you're missing a dependency. You should troubleshoot by
looking through the CMake output for any Java-related tools that aren't found and installing them.

![](images/cmake_output.png)

@note If CMake can't find Java in your system set the JAVA_HOME environment variable with the path to installed JDK before running it. E.g.:
@code{.bash}
export JAVA_HOME=/usr/lib/jvm/java-6-oracle
cmake -DBUILD_SHARED_LIBS=OFF ..
@endcode

Now start the build:
@code{.bash}
make -j8
@endcode
or
@code{.bat}
msbuild /m OpenCV.sln /t:Build /p:Configuration=Release /v:m
@endcode
Besides all this will create a jar containing the Java interface (`bin/opencv-244.jar`) and a native
dynamic library containing Java bindings and all the OpenCV stuff (`lib/libopencv_java244.so` or
`bin/Release/opencv_java244.dll` respectively). We'll use these files later.

Java sample with Ant
--------------------

@note The described sample is provided with OpenCV library in the `opencv/samples/java/ant`
folder.

-   Create a folder where you'll develop this sample application.

-   In this folder create the `build.xml` file with the following content using any text editor:
    @include samples/java/ant/build.xml
    @note This XML file can be reused for building other Java applications. It describes a common folder structure in the lines 3 - 12 and common targets for compiling and running the application.
    When reusing this XML don't forget to modify the project name in the line 1, that is also the
    name of the main class (line 14). The paths to OpenCV jar and jni lib are expected as parameters
    ("${ocvJarDir}" in line 5 and "${ocvLibDir}" in line 37), but you can hardcode these paths for
    your convenience. See [Ant documentation](http://ant.apache.org/manual/) for detailed
    description of its build file format.

-   Create an `src` folder next to the `build.xml` file and a `SimpleSample.java` file in it.

-   Put the following Java code into the `SimpleSample.java` file:
    @code{.java}
        import org.opencv.core.Core;
        import org.opencv.core.Mat;
        import org.opencv.core.CvType;
        import org.opencv.core.Scalar;

        class SimpleSample {

          static{ System.loadLibrary(Core.NATIVE_LIBRARY_NAME); }

          public static void main(String[] args) {
            System.out.println("Welcome to OpenCV " + Core.VERSION);
            Mat m = new Mat(5, 10, CvType.CV_8UC1, new Scalar(0));
            System.out.println("OpenCV Mat: " + m);
            Mat mr1 = m.row(1);
            mr1.setTo(new Scalar(1));
            Mat mc5 = m.col(5);
            mc5.setTo(new Scalar(5));
            System.out.println("OpenCV Mat data:\n" + m.dump());
          }

        }
        @endcode
-   Run the following command in console in the folder containing `build.xml`:
    @code{.bash}
    ant -DocvJarDir=path/to/dir/containing/opencv-244.jar -DocvLibDir=path/to/dir/containing/opencv_java244/native/library
    @endcode
    For example:
    @code{.bat}
    ant -DocvJarDir=X:\opencv-2.4.4\bin -DocvLibDir=X:\opencv-2.4.4\bin\Release
    @endcode
    The command should initiate [re]building and running the sample. You should see on the
    screen something like this:

    ![](images/ant_output.png)

SBT project for Java and Scala
------------------------------

Now we'll create a simple Java application using SBT. This serves as a brief introduction to those
unfamiliar with this build tool. We're using SBT because it is particularly easy and powerful.

First, download and install [SBT](http://www.scala-sbt.org/) using the instructions on its [web
site](http://www.scala-sbt.org/).

Next, navigate to a new directory where you'd like the application source to live (outside `opencv`
dir). Let's call it "JavaSample" and create a directory for it:
@code{.bash}
cd <somewhere outside opencv>
mkdir JavaSample
@endcode
Now we will create the necessary folders and an SBT project:
@code{.bash}
cd JavaSample
mkdir -p src/main/java # This is where SBT expects to find Java sources
mkdir project # This is where the build definitions live
@endcode
Now open `project/build.scala` in your favorite editor and paste the following. It defines your
project:
@code{.scala}
import sbt._
import Keys._

object JavaSampleBuild extends Build {
  def scalaSettings = Seq(
    scalaVersion := "2.10.0",
    scalacOptions ++= Seq(
      "-optimize",
      "-unchecked",
      "-deprecation"
    )
  )

  def buildSettings =
    Project.defaultSettings ++
    scalaSettings

  lazy val root = {
    val settings = buildSettings ++ Seq(name := "JavaSample")
    Project(id = "JavaSample", base = file("."), settings = settings)
  }
}
@endcode
Now edit `project/plugins.sbt` and paste the following. This will enable auto-generation of an
Eclipse project:
@code{.scala}
addSbtPlugin("com.typesafe.sbteclipse" % "sbteclipse-plugin" % "2.1.0")
@endcode
Now run sbt from the `JavaSample` root and from within SBT run eclipse to generate an eclipse
project:
@code{.bash}
sbt # Starts the sbt console
eclipse # Running "eclipse" from within the sbt console
@endcode
You should see something like this:

![](images/sbt_eclipse.png)

You can now import the SBT project to Eclipse using Import ... -\> Existing projects into workspace.
Whether you actually do this is optional for the guide; we'll be using SBT to build the project, so
if you choose to use Eclipse it will just serve as a text editor.

To test that everything is working, create a simple "Hello OpenCV" application. Do this by creating
a file `src/main/java/HelloOpenCV.java` with the following contents:
@code{.java}
public class HelloOpenCV {
  public static void main(String[] args) {
    System.out.println("Hello, OpenCV");
 }
@endcode
}

Now execute run from the sbt console, or more concisely, run sbt run from the command line:
@code{.bash}
sbt run
@endcode
You should see something like this:

![](images/sbt_run.png)

### Running SBT samples

Now we'll create a simple face detection application using OpenCV.

First, create a `lib/` folder and copy the OpenCV jar into it. By default, SBT adds jars in the lib
folder to the Java library search path. You can optionally rerun sbt eclipse to update your Eclipse
project.
@code{.bash}
mkdir lib
cp <opencv_dir>/build/bin/opencv_<version>.jar lib/
sbt eclipse
@endcode
Next, create the directory `src/main/resources` and download this Lena image into it:

![](images/lena.png)

Make sure it's called `"lena.png"`. Items in the resources directory are available to the Java
application at runtime.

Next, copy `lbpcascade_frontalface.xml` from `opencv/data/lbpcascades/` into the `resources`
directory:
@code{.bash}
cp <opencv_dir>/data/lbpcascades/lbpcascade_frontalface.xml src/main/resources/
@endcode
Now modify src/main/java/HelloOpenCV.java so it contains the following Java code:
@code{.java}
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.objdetect.CascadeClassifier;

//
// Detects faces in an image, draws boxes around them, and writes the results
// to "faceDetection.png".
//
class DetectFaceDemo {
  public void run() {
    System.out.println("\nRunning DetectFaceDemo");

    // Create a face detector from the cascade file in the resources
    // directory.
    CascadeClassifier faceDetector = new CascadeClassifier(getClass().getResource("/lbpcascade_frontalface.xml").getPath());
    Mat image = Imgcodecs.imread(getClass().getResource("/lena.png").getPath());

    // Detect faces in the image.
    // MatOfRect is a special container class for Rect.
    MatOfRect faceDetections = new MatOfRect();
    faceDetector.detectMultiScale(image, faceDetections);

    System.out.println(String.format("Detected %s faces", faceDetections.toArray().length));

    // Draw a bounding box around each face.
    for (Rect rect : faceDetections.toArray()) {
        Imgproc.rectangle(image, new Point(rect.x, rect.y), new Point(rect.x + rect.width, rect.y + rect.height), new Scalar(0, 255, 0));
    }

    // Save the visualized detection.
    String filename = "faceDetection.png";
    System.out.println(String.format("Writing %s", filename));
    Imgcodecs.imwrite(filename, image);
  }
}

public class HelloOpenCV {
  public static void main(String[] args) {
    System.out.println("Hello, OpenCV");

    // Load the native library.
    System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    new DetectFaceDemo().run();
  }
}
@endcode
Note the call to System.loadLibrary(Core.NATIVE_LIBRARY_NAME). This command must be executed
exactly once per Java process prior to using any native OpenCV methods. If you don't call it, you
will get UnsatisfiedLink errors. You will also get errors if you try to load OpenCV when it has
already been loaded.

Now run the face detection app using \`sbt run\`:
@code{.bash}
sbt run
@endcode
You should see something like this:

![](images/sbt_run_face.png)

It should also write the following image to `faceDetection.png`:

![](images/faceDetection.png)

You're done! Now you have a sample Java application working with OpenCV, so you can start the work
on your own. We wish you good luck and many years of joyful life!
