project(${the_module}_jar)

set(OPENCV_JAVA_DIR "${CMAKE_CURRENT_BINARY_DIR}/opencv" CACHE INTERNAL "")

file(REMOVE_RECURSE "${OPENCV_JAVA_DIR}")
file(REMOVE "${OPENCV_DEPHELPER}/${the_module}_jar_source_copy")

set(java_src_dir "${OPENCV_JAVA_DIR}/java")
file(MAKE_DIRECTORY "${java_src_dir}")

set(JAR_NAME_WE opencv-${OPENCV_JAVA_LIB_NAME_SUFFIX})
set(JAR_NAME ${JAR_NAME_WE}.jar)
set(OPENCV_JAR_DIR "${OpenCV_BINARY_DIR}/bin/" CACHE INTERNAL "")
set(OPENCV_JAR_FILE "${OPENCV_JAR_DIR}${JAR_NAME}" CACHE INTERNAL "")
set(OPENCV_SRC_JAR_FILE "${OpenCV_BINARY_DIR}/bin/opencv-${OPENCV_JAVA_LIB_NAME_SUFFIX}-sources.jar" CACHE INTERNAL "")

ocv_copyfiles_append_dir(JAVA_SRC_COPY "${OPENCV_JAVA_BINDINGS_DIR}/gen/java" "${java_src_dir}")

set(depends gen_opencv_java_source "${OPENCV_DEPHELPER}/gen_opencv_java_source")
ocv_copyfiles_add_target(${the_module}_jar_source_copy JAVA_SRC_COPY "Copy Java(JAR) source files" ${depends})
set(depends ${the_module}_jar_source_copy "${OPENCV_DEPHELPER}/${the_module}_jar_source_copy")

if(OPENCV_JAVA_SOURCE_VERSION)
  set(OPENCV_ANT_JAVAC_EXTRA_ATTRS "${OPENCV_ANT_JAVAC_EXTRA_ATTRS} source=\"${OPENCV_JAVA_SOURCE_VERSION}\"")
endif()
if(OPENCV_JAVA_TARGET_VERSION)
  set(OPENCV_ANT_JAVAC_EXTRA_ATTRS "${OPENCV_ANT_JAVAC_EXTRA_ATTRS} target=\"${OPENCV_JAVA_TARGET_VERSION}\"")
endif()

set(OPENCV_JAVADOC_DESTINATION "${OpenCV_BINARY_DIR}/doc/doxygen/html/javadoc" CACHE STRING "")

# Old Javadoc URL looks like this: https://docs.oracle.com/javase/6/docs/api/
# New Javadoc URL looks like this: https://docs.oracle.com/en/java/javase/11/docs/api/
set(OPENCV_JAVADOC_LINK_URL "" CACHE STRING "See details in modules/java/jar/CMakeLists.txt")
if(OPENCV_JAVADOC_LINK_URL)
  set(CMAKE_CONFIG_OPENCV_JAVADOC_LINK "link=\"${OPENCV_JAVADOC_LINK_URL}\"")
endif()

if(OPENCV_JAVA_SDK_BUILD_TYPE STREQUAL "ANT")
  file(MAKE_DIRECTORY "${OPENCV_JAVA_DIR}/build/classes")

  configure_file("${CMAKE_CURRENT_SOURCE_DIR}/build.xml.in" "${OPENCV_JAVA_DIR}/build.xml" @ONLY)
  list(APPEND depends "${OPENCV_JAVA_DIR}/build.xml")

  ocv_cmake_byproducts(__byproducts BYPRODUCTS "${OPENCV_JAR_FILE}")
  add_custom_command(OUTPUT "${OPENCV_DEPHELPER}/${the_module}_jar"
      ${__byproducts}  # required for add_custom_target() by ninja
      COMMAND ${ANT_EXECUTABLE} -noinput -k jar
      COMMAND ${CMAKE_COMMAND} -E touch "${OPENCV_DEPHELPER}/${the_module}_jar"
      WORKING_DIRECTORY "${OPENCV_JAVA_DIR}"
      DEPENDS ${depends}
      COMMENT "Generating ${JAR_NAME}"
  )
  add_custom_target(${the_module}_jar DEPENDS "${OPENCV_DEPHELPER}/${the_module}_jar")
elseif(OPENCV_JAVA_SDK_BUILD_TYPE STREQUAL "JAVA")
  configure_file("${CMAKE_CURRENT_SOURCE_DIR}/MANIFEST.MF.in" "${OPENCV_JAVA_DIR}/MANIFEST.MF" @ONLY)
  list(APPEND depends "${OPENCV_JAVA_DIR}/MANIFEST.MF")

  ocv_cmake_byproducts(__byproducts BYPRODUCTS "${OPENCV_JAVA_DIR}/java_sources")
  add_custom_command(OUTPUT "${OPENCV_DEPHELPER}/${the_module}_jar"
      BYPRODUCTS ${__byproducts}  # required for add_custom_target() by ninja
      DEPENDS ${depends}
      COMMAND ${CMAKE_COMMAND} -E touch "${OPENCV_DEPHELPER}/${the_module}_jar"
      COMMAND ${CMAKE_COMMAND}
      -D OPENCV_JAVA_DIR="${OPENCV_JAVA_DIR}/java"
      -D OUTPUT="${OPENCV_JAVA_DIR}/java_sources"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/list_java_sources.cmake"
  )

  add_custom_target(${the_module}_jar_sources
    DEPENDS "${OPENCV_DEPHELPER}/${the_module}_jar"
  )

  list(APPEND CMAKE_JAVA_COMPILE_FLAGS -encoding utf-8 ${OPENCV_EXTRA_JAVA_COMPILE_FLAGS})

  add_jar(${the_module}_jar
          SOURCES "@${OPENCV_JAVA_DIR}/java_sources"
          MANIFEST "${OPENCV_JAVA_DIR}/MANIFEST.MF"
          OUTPUT_NAME "${JAR_NAME_WE}"
          OUTPUT_DIR "${OPENCV_JAR_DIR}")

  add_dependencies(${the_module}_jar ${the_module}_jar_sources)
else()
  ocv_assert(0)
endif()

install(FILES ${OPENCV_JAR_FILE} OPTIONAL DESTINATION ${OPENCV_JAR_INSTALL_PATH} COMPONENT java)

add_dependencies(${the_module} ${the_module}_jar)

if(BUILD_DOCS)
  if(OPENCV_JAVA_SDK_BUILD_TYPE STREQUAL "ANT")
    add_custom_command(OUTPUT "${OPENCV_DEPHELPER}/${the_module}doc"
      COMMAND ${ANT_EXECUTABLE} -noinput -k javadoc
      COMMAND ${CMAKE_COMMAND} -E touch "${OPENCV_DEPHELPER}/${the_module}doc"
      WORKING_DIRECTORY "${OPENCV_JAVA_DIR}"
      DEPENDS ${depends}
      COMMENT "Generating Javadoc"
    )
    add_custom_target(${the_module}doc DEPENDS "${OPENCV_DEPHELPER}/${the_module}doc")

    install(DIRECTORY ${OpenCV_BINARY_DIR}/doc/doxygen/html/javadoc
      DESTINATION "${OPENCV_DOC_INSTALL_PATH}/html"
      COMPONENT "docs" OPTIONAL
      ${compatible_MESSAGE_NEVER}
    )
  elseif(OPENCV_JAVA_SDK_BUILD_TYPE STREQUAL "JAVA")
    set(Java_JAVADOC_EXECUTABLE ${Java_JAVADOC_EXECUTABLE} -encoding utf-8)

    # create_javadoc produces target ${_target}_javadoc
    create_javadoc(${the_module}
      FILES "@${OPENCV_JAVA_DIR}/java_sources"
      SOURCEPATH "${OPENCV_JAVA_DIR}/java"
      INSTALLPATH "${OPENCV_JAVADOC_DESTINATION}"
      WINDOWTITLE "OpenCV ${OPENCV_VERSION_PLAIN} Java documentation"
      DOCTITLE "OpenCV Java documentation (${OPENCV_VERSION})"
      VERSION TRUE
    )
    add_dependencies(${the_module}_javadoc ${the_module}_jar_sources)
    add_custom_target(${the_module}doc DEPENDS ${the_module}_javadoc)

    install(DIRECTORY ${OpenCV_BINARY_DIR}/doc/doxygen/html/javadoc/${the_module}/
      DESTINATION "${OPENCV_DOC_INSTALL_PATH}/html/javadoc"
      COMPONENT "docs" OPTIONAL
      ${compatible_MESSAGE_NEVER}
    )
  else()
    ocv_assert(0)
  endif()

  set(CMAKE_DOXYGEN_JAVADOC_NODE
    "<tab type=\"user\" url=\"./javadoc/index.html\" title=\"Java documentation\"/>"
    CACHE INTERNAL "Link to the Java documentation") # set to the cache to make it global
  add_custom_target(doxygen_javadoc DEPENDS ${the_module}doc)
  add_dependencies(opencv_docs ${the_module}doc)
else()
  unset(CMAKE_DOXYGEN_JAVADOC_NODE CACHE)
endif()
