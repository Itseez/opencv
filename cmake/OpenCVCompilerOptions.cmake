if("${CMAKE_CXX_COMPILER};${CMAKE_C_COMPILER};${CMAKE_CXX_COMPILER_LAUNCHER}" MATCHES "ccache")
  set(OPENCV_COMPILER_IS_CCACHE 1)
endif()
if(ENABLE_CCACHE AND NOT OPENCV_COMPILER_IS_CCACHE)
  # This works fine with Unix Makefiles and Ninja generators
  find_host_program(CCACHE_PROGRAM ccache)
  if(CCACHE_PROGRAM)
    message(STATUS "Looking for ccache - found (${CCACHE_PROGRAM})")
    get_property(__OLD_RULE_LAUNCH_COMPILE GLOBAL PROPERTY RULE_LAUNCH_COMPILE)
    if(CMAKE_GENERATOR MATCHES "Xcode")
      configure_file("${CMAKE_CURRENT_LIST_DIR}/templates/xcode-launch-c.in" "${CMAKE_BINARY_DIR}/xcode-launch-c")
      configure_file("${CMAKE_CURRENT_LIST_DIR}/templates/xcode-launch-cxx.in" "${CMAKE_BINARY_DIR}/xcode-launch-cxx")
      execute_process(COMMAND chmod a+rx
          "${CMAKE_BINARY_DIR}/xcode-launch-c"
          "${CMAKE_BINARY_DIR}/xcode-launch-cxx"
      )
      # Xcode project attributes
      set(CMAKE_XCODE_ATTRIBUTE_CC         "${CMAKE_BINARY_DIR}/xcode-launch-c")
      set(CMAKE_XCODE_ATTRIBUTE_CXX        "${CMAKE_BINARY_DIR}/xcode-launch-cxx")
      set(CMAKE_XCODE_ATTRIBUTE_LD         "${CMAKE_BINARY_DIR}/xcode-launch-c")
      set(CMAKE_XCODE_ATTRIBUTE_LDPLUSPLUS "${CMAKE_BINARY_DIR}/xcode-launch-cxx")
      set(OPENCV_COMPILER_IS_CCACHE 1)
      message(STATUS "ccache: enable support through Xcode project properties")
    elseif(__OLD_RULE_LAUNCH_COMPILE)
      message(STATUS "Can't replace CMake compiler launcher")
    else()
      set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CCACHE_PROGRAM}")
      # NOTE: Actually this check doesn't work as expected.
      # "RULE_LAUNCH_COMPILE" is ignored by CMake during try_compile() step.
      # ocv_check_compiler_flag(CXX "" IS_CCACHE_WORKS)
      set(IS_CCACHE_WORKS 1)
      if(IS_CCACHE_WORKS)
        set(OPENCV_COMPILER_IS_CCACHE 1)
      else()
        message(STATUS "Unable to compile program with enabled ccache, reverting...")
        set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${__OLD_RULE_LAUNCH_COMPILE}")
      endif()
    endif()
  else()
    message(STATUS "Looking for ccache - not found")
  endif()
endif()

if((CV_CLANG AND NOT CMAKE_GENERATOR MATCHES "Xcode")  # PCH has no support for Clang
    OR OPENCV_COMPILER_IS_CCACHE
)
  set(ENABLE_PRECOMPILED_HEADERS OFF CACHE BOOL "" FORCE)
endif()

macro(add_extra_compiler_option option)
  ocv_check_flag_support(CXX "${option}" _varname "${OPENCV_EXTRA_CXX_FLAGS} ${ARGN}")
  if(${_varname})
    set(OPENCV_EXTRA_CXX_FLAGS "${OPENCV_EXTRA_CXX_FLAGS} ${option}")
  endif()

  ocv_check_flag_support(C "${option}" _varname "${OPENCV_EXTRA_C_FLAGS} ${ARGN}")
  if(${_varname})
    set(OPENCV_EXTRA_C_FLAGS "${OPENCV_EXTRA_C_FLAGS} ${option}")
  endif()
endmacro()

macro(add_extra_compiler_option_force option)
  set(OPENCV_EXTRA_CXX_FLAGS "${OPENCV_EXTRA_CXX_FLAGS} ${option}")
  set(OPENCV_EXTRA_C_FLAGS "${OPENCV_EXTRA_C_FLAGS} ${option}")
endmacro()


# Gets environment variable and puts its value to the corresponding preprocessor definition
# Useful for WINRT that has no access to environment variables
macro(add_env_definitions option)
  set(value $ENV{${option}})
  if("${value}" STREQUAL "")
    message(WARNING "${option} environment variable is empty. Please set it to appropriate location to get correct results")
  else()
    string(REPLACE "\\" "\\\\" value ${value})
  endif()
  add_definitions("-D${option}=\"${value}\"")
endmacro()

# Use same flags for native AArch64 and RISC-V compilation as for cross-compile (Linux)
if(NOT CMAKE_CROSSCOMPILING AND NOT CMAKE_TOOLCHAIN_FILE AND COMMAND ocv_set_platform_flags)
  unset(platform_flags)
  ocv_set_platform_flags(platform_flags)
  # externally-provided flags should have higher priority - prepend our flags
  if(platform_flags)
    set(CMAKE_CXX_FLAGS "${platform_flags} ${CMAKE_CXX_FLAGS}")
    set(CMAKE_C_FLAGS "${platform_flags} ${CMAKE_C_FLAGS}")
  endif()
endif()

if(NOT MSVC)
  # OpenCV fails some tests when 'char' is 'unsigned' by default
  add_extra_compiler_option(-fsigned-char)
endif()

if(MSVC)
  if(NOT " ${CMAKE_CXX_FLAGS} ${OPENCV_EXTRA_FLAGS} ${OPENCV_EXTRA_CXX_FLAGS}" MATCHES " /fp:")
    if(ENABLE_FAST_MATH)
      add_extra_compiler_option("/fp:fast")
    else()
      add_extra_compiler_option("/fp:precise")
    endif()
  endif()
elseif(CV_ICC)
  if(NOT " ${CMAKE_CXX_FLAGS} ${OPENCV_EXTRA_FLAGS} ${OPENCV_EXTRA_CXX_FLAGS}" MATCHES " /fp:"
      AND NOT " ${CMAKE_CXX_FLAGS} ${OPENCV_EXTRA_FLAGS} ${OPENCV_EXTRA_CXX_FLAGS}" MATCHES " -fp-model"
  )
    if(NOT ENABLE_FAST_MATH)
      add_extra_compiler_option("-fp-model precise")
    endif()
  endif()
elseif(CV_ICX)
  # ICX uses -ffast-math by default.
  # use own flags, if no one of the flags provided by user: -fp-model, -ffast-math -fno-fast-math
  if(NOT " ${CMAKE_CXX_FLAGS} ${OPENCV_EXTRA_FLAGS} ${OPENCV_EXTRA_CXX_FLAGS}" MATCHES " /fp:"
      AND NOT " ${CMAKE_CXX_FLAGS} ${OPENCV_EXTRA_FLAGS} ${OPENCV_EXTRA_CXX_FLAGS}" MATCHES " -fp-model"
      AND NOT " ${CMAKE_CXX_FLAGS} ${OPENCV_EXTRA_FLAGS} ${OPENCV_EXTRA_CXX_FLAGS}" MATCHES " -ffast-math"
      AND NOT " ${CMAKE_CXX_FLAGS} ${OPENCV_EXTRA_FLAGS} ${OPENCV_EXTRA_CXX_FLAGS}" MATCHES " -fno-fast-math"
  )
    if(NOT ENABLE_FAST_MATH)
      add_extra_compiler_option(-fno-fast-math)
      add_extra_compiler_option(-fp-model=precise)
    endif()
  endif()
elseif(CV_GCC OR CV_CLANG)
  if(ENABLE_FAST_MATH)
    add_extra_compiler_option(-ffast-math)
    add_extra_compiler_option(-fno-finite-math-only)
  endif()
endif()

if(CV_GCC OR CV_CLANG OR CV_ICX)
  # High level of warnings.
  add_extra_compiler_option(-W)
  if (NOT MSVC)
    # clang-cl interprets -Wall as MSVC would: -Weverything, which is more than
    # we want.
    add_extra_compiler_option(-Wall)
  endif()
  add_extra_compiler_option(-Wreturn-type)
  add_extra_compiler_option(-Wnon-virtual-dtor)
  add_extra_compiler_option(-Waddress)
  add_extra_compiler_option(-Wsequence-point)
  add_extra_compiler_option(-Wformat)
  add_extra_compiler_option(-Wformat-security -Wformat)
  add_extra_compiler_option(-Wmissing-declarations)
  add_extra_compiler_option(-Wmissing-prototypes)
  add_extra_compiler_option(-Wstrict-prototypes)
  add_extra_compiler_option(-Wundef)
  add_extra_compiler_option(-Winit-self)
  add_extra_compiler_option(-Wpointer-arith)
  if(NOT (CV_GCC AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS "5.0"))
    add_extra_compiler_option(-Wshadow)  # old GCC emits warnings for variables + methods combination
  endif()
  add_extra_compiler_option(-Wsign-promo)
  add_extra_compiler_option(-Wuninitialized)
  if(CV_GCC AND (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 6.0) AND (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 7.0 OR ARM))
    add_extra_compiler_option(-Wno-psabi)
  endif()
  if(HAVE_CXX11)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND NOT ENABLE_PRECOMPILED_HEADERS)
      add_extra_compiler_option(-Wsuggest-override)
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
      add_extra_compiler_option(-Winconsistent-missing-override)
    endif()
  endif()

  if(ENABLE_NOISY_WARNINGS)
    add_extra_compiler_option(-Wcast-align)
    add_extra_compiler_option(-Wstrict-aliasing=2)
  else()
    add_extra_compiler_option(-Wno-delete-non-virtual-dtor)
    add_extra_compiler_option(-Wno-unnamed-type-template-args)
    add_extra_compiler_option(-Wno-comment)
    if(NOT OPENCV_SKIP_IMPLICIT_FALLTHROUGH
        AND NOT " ${CMAKE_CXX_FLAGS} ${OPENCV_EXTRA_FLAGS} ${OPENCV_EXTRA_CXX_FLAGS}" MATCHES "implicit-fallthrough"
        AND (CV_GCC AND NOT CMAKE_CXX_COMPILER_VERSION VERSION_LESS 7.0.0)
    )
      add_extra_compiler_option(-Wimplicit-fallthrough=3)
    endif()
    if(CV_GCC AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 7.0)
      add_extra_compiler_option(-Wno-strict-overflow) # Issue appears when compiling surf.cpp from opencv_contrib/modules/xfeatures2d
    endif()
    if(CV_GCC AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 5.0)
      add_extra_compiler_option(-Wno-missing-field-initializers)  # GCC 4.x emits warnings about {}, fixed in GCC 5+
    endif()
    if(CV_CLANG AND NOT CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang" AND NOT CMAKE_CXX_COMPILER_VERSION VERSION_LESS 10.0)
      add_extra_compiler_option(-Wno-deprecated-enum-enum-conversion)
      add_extra_compiler_option(-Wno-deprecated-anon-enum-enum-conversion)
    endif()
  endif()
  add_extra_compiler_option(-fdiagnostics-show-option)

  # We need pthread's, unless we have explicitly disabled multi-thread execution.
  if(NOT OPENCV_DISABLE_THREAD_SUPPORT
      AND (
        (UNIX
          AND NOT ANDROID
          AND NOT (APPLE AND CV_CLANG)
          AND NOT EMSCRIPTEN
        )
        OR (EMSCRIPTEN AND WITH_PTHREADS_PF)  # https://github.com/opencv/opencv/issues/20285
      )
  ) # TODO
    add_extra_compiler_option(-pthread)
  endif()

  if(CV_CLANG)
    add_extra_compiler_option(-Qunused-arguments)
  endif()

  if(OPENCV_WARNINGS_ARE_ERRORS)
    add_extra_compiler_option(-Werror)
  endif()

  if(APPLE)
    add_extra_compiler_option(-Wno-semicolon-before-method-body)
  endif()

  # Other optimizations
  if(ENABLE_OMIT_FRAME_POINTER)
    add_extra_compiler_option(-fomit-frame-pointer)
  elseif(DEFINED ENABLE_OMIT_FRAME_POINTER)
    add_extra_compiler_option(-fno-omit-frame-pointer)
  endif()

  # Profiling?
  if(ENABLE_PROFILING)
    # turn off incompatible options
    foreach(flags CMAKE_CXX_FLAGS CMAKE_C_FLAGS CMAKE_CXX_FLAGS_RELEASE CMAKE_C_FLAGS_RELEASE CMAKE_CXX_FLAGS_DEBUG CMAKE_C_FLAGS_DEBUG
                  OPENCV_EXTRA_FLAGS_RELEASE OPENCV_EXTRA_FLAGS_DEBUG OPENCV_EXTRA_C_FLAGS OPENCV_EXTRA_CXX_FLAGS)
      string(REPLACE "-fomit-frame-pointer" "" ${flags} "${${flags}}")
      string(REPLACE "-ffunction-sections" "" ${flags} "${${flags}}")
      string(REPLACE "-fdata-sections" "" ${flags} "${${flags}}")
    endforeach()
    # -pg should be placed both in the linker and in the compiler settings
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pg")
    add_extra_compiler_option("-pg -g")
  else()
    if(MSVC)
      # TODO: Clang/C2 is not supported
    elseif(((IOS OR ANDROID) AND NOT BUILD_SHARED_LIBS) AND NOT OPENCV_FORCE_FUNCTIONS_SECTIONS)
      # don't create separate sections for functions/data, reduce package size
    else()
      # Remove unreferenced functions: function level linking
      add_extra_compiler_option(-ffunction-sections)
      add_extra_compiler_option(-fdata-sections)
      if(NOT OPENCV_SKIP_GC_SECTIONS)
        if(APPLE)
          set(OPENCV_EXTRA_EXE_LINKER_FLAGS "${OPENCV_EXTRA_EXE_LINKER_FLAGS} -Wl,-dead_strip")
          set(OPENCV_EXTRA_SHARED_LINKER_FLAGS "${OPENCV_EXTRA_SHARED_LINKER_FLAGS} -Wl,-dead_strip")
          set(OPENCV_EXTRA_MODULE_LINKER_FLAGS "${OPENCV_EXTRA_MODULE_LINKER_FLAGS} -Wl,-dead_strip")
        else()
          set(OPENCV_EXTRA_EXE_LINKER_FLAGS "${OPENCV_EXTRA_EXE_LINKER_FLAGS} -Wl,--gc-sections")
          set(OPENCV_EXTRA_SHARED_LINKER_FLAGS "${OPENCV_EXTRA_SHARED_LINKER_FLAGS} -Wl,--gc-sections")
          set(OPENCV_EXTRA_MODULE_LINKER_FLAGS "${OPENCV_EXTRA_MODULE_LINKER_FLAGS} -Wl,--gc-sections")
        endif()
      endif()
    endif()
  endif()

  if(ENABLE_COVERAGE)
    set(OPENCV_EXTRA_C_FLAGS "${OPENCV_EXTRA_C_FLAGS} --coverage")
    set(OPENCV_EXTRA_CXX_FLAGS "${OPENCV_EXTRA_CXX_FLAGS} --coverage")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} --coverage")
  endif()

  if(ENABLE_INSTRUMENTATION)
    if(NOT HAVE_CXX11)
      message(WARNING "ENABLE_INSTRUMENTATION requires C++11 support")
    endif()
    set(WITH_VTK OFF) # There are issues with VTK 6.0
  endif()

  if(ENABLE_LTO)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 12)
      add_extra_compiler_option(-flto=auto)
    else()
      add_extra_compiler_option(-flto)
    endif()
  endif()
  if(ENABLE_THIN_LTO)
    add_extra_compiler_option(-flto=thin)
  endif()

  set(OPENCV_EXTRA_FLAGS_RELEASE "${OPENCV_EXTRA_FLAGS_RELEASE} -DNDEBUG")
  if(NOT " ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG} ${OPENCV_EXTRA_FLAGS_DEBUG} " MATCHES "-O")
    set(OPENCV_EXTRA_FLAGS_DEBUG "${OPENCV_EXTRA_FLAGS_DEBUG} -O0")
  endif()
  set(OPENCV_EXTRA_FLAGS_DEBUG "${OPENCV_EXTRA_FLAGS_DEBUG} -DDEBUG -D_DEBUG")

  if(BUILD_WITH_DEBUG_INFO)
    # https://gcc.gnu.org/onlinedocs/gcc/Debugging-Options.html
    # '-g' is equal to '-g2', '-g1' produces minimal information, enough for making backtraces
    ocv_update(OPENCV_DEBUG_OPTION "-g1")
    if(NOT " ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE} ${OPENCV_EXTRA_CXX_FLAGS} ${OPENCV_EXTRA_FLAGS_RELEASE}" MATCHES " -g")
      set(OPENCV_EXTRA_FLAGS_RELEASE "${OPENCV_EXTRA_FLAGS_RELEASE} ${OPENCV_DEBUG_OPTION}")
    endif()
  endif()
endif()

if(MSVC)
  #TODO Code refactoring is required to resolve security warnings
  #if(NOT ENABLE_BUILD_HARDENING)
    set(OPENCV_EXTRA_FLAGS "${OPENCV_EXTRA_FLAGS} /D _CRT_SECURE_NO_DEPRECATE /D _CRT_NONSTDC_NO_DEPRECATE /D _SCL_SECURE_NO_WARNINGS")
  #endif()

  if(BUILD_WITH_DEBUG_INFO)
    set(OPENCV_EXTRA_FLAGS_RELEASE "${OPENCV_EXTRA_FLAGS_RELEASE} /Zi")
    set(OPENCV_EXTRA_EXE_LINKER_FLAGS_RELEASE "${OPENCV_EXTRA_EXE_LINKER_FLAGS_RELEASE} /debug")
    set(OPENCV_EXTRA_SHARED_LINKER_FLAGS_RELEASE "${OPENCV_EXTRA_SHARED_LINKER_FLAGS_RELEASE} /debug")
    set(OPENCV_EXTRA_MODULE_LINKER_FLAGS_RELEASE "${OPENCV_EXTRA_MODULE_LINKER_FLAGS_RELEASE} /debug")
  endif()

  # Remove unreferenced functions: function level linking
  set(OPENCV_EXTRA_FLAGS "${OPENCV_EXTRA_FLAGS} /Gy")
  if(NOT MSVC_VERSION LESS 1400)
    set(OPENCV_EXTRA_FLAGS "${OPENCV_EXTRA_FLAGS} /bigobj")
  endif()

  if(OPENCV_WARNINGS_ARE_ERRORS)
    set(OPENCV_EXTRA_FLAGS "${OPENCV_EXTRA_FLAGS} /WX")
  endif()

  if(ENABLE_LTO)
    set(OPENCV_EXTRA_FLAGS_RELEASE "${OPENCV_EXTRA_FLAGS_RELEASE} /GL")
    set(OPENCV_EXTRA_EXE_LINKER_FLAGS_RELEASE "${OPENCV_EXTRA_EXE_LINKER_FLAGS_RELEASE} /LTCG")
  endif()

  if(NOT MSVC_VERSION LESS 1800 AND NOT CMAKE_GENERATOR MATCHES "Visual Studio")
    set(OPENCV_EXTRA_C_FLAGS "${OPENCV_EXTRA_C_FLAGS} /FS")
    set(OPENCV_EXTRA_CXX_FLAGS "${OPENCV_EXTRA_CXX_FLAGS} /FS")
  endif()

  if(AARCH64 AND NOT MSVC_VERSION LESS 1930)
    set(OPENCV_EXTRA_FLAGS "${OPENCV_EXTRA_FLAGS} /D _ARM64_DISTINCT_NEON_TYPES")
  endif()
endif()

if(PROJECT_NAME STREQUAL "OpenCV")
  include("${OpenCV_SOURCE_DIR}/cmake/OpenCVCompilerOptimizations.cmake")
endif()
if(COMMAND ocv_compiler_optimization_options)
  ocv_compiler_optimization_options()
endif()
if(COMMAND ocv_compiler_optimization_options_finalize)
  ocv_compiler_optimization_options_finalize()
endif()

# set default visibility to hidden
if((CV_GCC OR CV_CLANG OR CV_ICX)
    AND NOT MSVC
    AND NOT OPENCV_SKIP_VISIBILITY_HIDDEN
    AND NOT " ${CMAKE_CXX_FLAGS} ${OPENCV_EXTRA_FLAGS} ${OPENCV_EXTRA_CXX_FLAGS}" MATCHES " -fvisibility")
  add_extra_compiler_option(-fvisibility=hidden)
  add_extra_compiler_option(-fvisibility-inlines-hidden)
endif()

# workaround gcc bug for aligned ld/st
# https://github.com/opencv/opencv/issues/13211
if((PPC64LE AND NOT CMAKE_CROSSCOMPILING) OR OPENCV_FORCE_COMPILER_CHECK_VSX_ALIGNED)
  ocv_check_runtime_flag("${CPU_BASELINE_FLAGS}" OPENCV_CHECK_VSX_ALIGNED "${OpenCV_SOURCE_DIR}/cmake/checks/runtime/cpu_vsx_aligned.cpp")
  if(NOT OPENCV_CHECK_VSX_ALIGNED)
    add_extra_compiler_option_force(-DCV_COMPILER_VSX_BROKEN_ALIGNED)
  endif()
endif()
# validate inline asm with fixes register number and constraints wa, wd, wf
if(PPC64LE)
  ocv_check_compiler_flag(CXX "${CPU_BASELINE_FLAGS}" OPENCV_CHECK_VSX_ASM "${OpenCV_SOURCE_DIR}/cmake/checks/cpu_vsx_asm.cpp")
  if(NOT OPENCV_CHECK_VSX_ASM)
    add_extra_compiler_option_force(-DCV_COMPILER_VSX_BROKEN_ASM)
  endif()
endif()

# Apply "-Wl,--as-needed" linker flags: https://github.com/opencv/opencv/issues/7001
if(NOT OPENCV_SKIP_LINK_AS_NEEDED)
  if(UNIX)
    set(_option "-Wl,--as-needed")
    set(_saved_CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${_option}")  # requires CMake 3.2+ and CMP0056
    ocv_check_compiler_flag(CXX "" HAVE_LINK_AS_NEEDED)
    set(CMAKE_EXE_LINKER_FLAGS "${_saved_CMAKE_EXE_LINKER_FLAGS}")
    if(HAVE_LINK_AS_NEEDED)
      set(OPENCV_EXTRA_EXE_LINKER_FLAGS "${OPENCV_EXTRA_EXE_LINKER_FLAGS} ${_option}")
      set(OPENCV_EXTRA_SHARED_LINKER_FLAGS "${OPENCV_EXTRA_SHARED_LINKER_FLAGS} ${_option}")
      set(OPENCV_EXTRA_MODULE_LINKER_FLAGS "${OPENCV_EXTRA_MODULE_LINKER_FLAGS} ${_option}")
    endif()
  endif()
endif()

# Apply "-Wl,--no-undefined" linker flags: https://github.com/opencv/opencv/pull/21347
if(NOT OPENCV_SKIP_LINK_NO_UNDEFINED)
  if(UNIX AND (NOT CMAKE_SYSTEM_NAME MATCHES "OpenBSD"))
    set(_option "-Wl,--no-undefined")
    set(_saved_CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${_option}")  # requires CMake 3.2+ and CMP0056
    ocv_check_compiler_flag(CXX "" HAVE_LINK_NO_UNDEFINED)
    set(CMAKE_EXE_LINKER_FLAGS "${_saved_CMAKE_EXE_LINKER_FLAGS}")
    if(HAVE_LINK_NO_UNDEFINED)
      set(OPENCV_EXTRA_EXE_LINKER_FLAGS "${OPENCV_EXTRA_EXE_LINKER_FLAGS} ${_option}")
      set(OPENCV_EXTRA_SHARED_LINKER_FLAGS "${OPENCV_EXTRA_SHARED_LINKER_FLAGS} ${_option}")
      set(OPENCV_EXTRA_MODULE_LINKER_FLAGS "${OPENCV_EXTRA_MODULE_LINKER_FLAGS} ${_option}")
    endif()
  endif()
endif()

# For 16k pages support with NDK prior 27
# Details: https://developer.android.com/guide/practices/page-sizes?hl=en
if(ANDROID AND ANDROID_SUPPORT_FLEXIBLE_PAGE_SIZES AND (ANDROID_ABI STREQUAL arm64-v8a OR ANDROID_ABI STREQUAL x86_64))
 set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-z,max-page-size=16384")
 set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-z,max-page-size=16384")
 set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} -Wl,-z,max-page-size=16384")
endif()

# combine all "extra" options
if(NOT OPENCV_SKIP_EXTRA_COMPILER_FLAGS)
  set(CMAKE_C_FLAGS           "${CMAKE_C_FLAGS} ${OPENCV_EXTRA_FLAGS} ${OPENCV_EXTRA_C_FLAGS}")
  set(CMAKE_CXX_FLAGS         "${CMAKE_CXX_FLAGS} ${OPENCV_EXTRA_FLAGS} ${OPENCV_EXTRA_CXX_FLAGS}")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${OPENCV_EXTRA_FLAGS_RELEASE}")
  set(CMAKE_C_FLAGS_RELEASE   "${CMAKE_C_FLAGS_RELEASE} ${OPENCV_EXTRA_FLAGS_RELEASE}")
  set(CMAKE_CXX_FLAGS_DEBUG   "${CMAKE_CXX_FLAGS_DEBUG} ${OPENCV_EXTRA_FLAGS_DEBUG}")
  set(CMAKE_C_FLAGS_DEBUG     "${CMAKE_C_FLAGS_DEBUG} ${OPENCV_EXTRA_FLAGS_DEBUG}")
  set(CMAKE_EXE_LINKER_FLAGS         "${CMAKE_EXE_LINKER_FLAGS} ${OPENCV_EXTRA_EXE_LINKER_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} ${OPENCV_EXTRA_EXE_LINKER_FLAGS_RELEASE}")
  set(CMAKE_EXE_LINKER_FLAGS_DEBUG   "${CMAKE_EXE_LINKER_FLAGS_DEBUG} ${OPENCV_EXTRA_EXE_LINKER_FLAGS_DEBUG}")
  set(CMAKE_SHARED_LINKER_FLAGS         "${CMAKE_SHARED_LINKER_FLAGS} ${OPENCV_EXTRA_SHARED_LINKER_FLAGS}")
  set(CMAKE_SHARED_LINKER_FLAGS_RELEASE "${CMAKE_SHARED_LINKER_FLAGS_RELEASE} ${OPENCV_EXTRA_SHARED_LINKER_FLAGS_RELEASE}")
  set(CMAKE_SHARED_LINKER_FLAGS_DEBUG   "${CMAKE_SHARED_LINKER_FLAGS_DEBUG} ${OPENCV_EXTRA_SHARED_LINKER_FLAGS_DEBUG}")
  set(CMAKE_MODULE_LINKER_FLAGS         "${CMAKE_MODULE_LINKER_FLAGS} ${OPENCV_EXTRA_MODULE_LINKER_FLAGS}")
  set(CMAKE_MODULE_LINKER_FLAGS_RELEASE "${CMAKE_MODULE_LINKER_FLAGS_RELEASE} ${OPENCV_EXTRA_MODULE_LINKER_FLAGS_RELEASE}")
  set(CMAKE_MODULE_LINKER_FLAGS_DEBUG   "${CMAKE_MODULE_LINKER_FLAGS_DEBUG} ${OPENCV_EXTRA_MODULE_LINKER_FLAGS_DEBUG}")
endif()

if(MSVC)
  if(NOT ENABLE_NOISY_WARNINGS)
    if(MSVC_VERSION EQUAL 1400)
      ocv_warnings_disable(CMAKE_CXX_FLAGS /wd4510 /wd4610 /wd4312 /wd4201 /wd4244 /wd4328 /wd4267)
    endif()
  endif()

  foreach(flags CMAKE_C_FLAGS CMAKE_C_FLAGS_RELEASE CMAKE_C_FLAGS_RELEASE CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_RELEASE CMAKE_CXX_FLAGS_DEBUG)
    string(REPLACE "/Zm1000" "" ${flags} "${${flags}}")
  endforeach()

  # Enable 'extern "C"' and asynchronous (division by zero, access violation) exceptions
  if(NOT OPENCV_SKIP_MSVC_EXCEPTIONS_FLAG)
    foreach(flags CMAKE_C_FLAGS CMAKE_C_FLAGS_RELEASE CMAKE_C_FLAGS_RELEASE CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_RELEASE CMAKE_CXX_FLAGS_DEBUG)
      string(REGEX REPLACE " /EH[^ ]* " " " ${flags} " ${${flags}}")
    endforeach()
    if(NOT " ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE} ${CMAKE_CXX_FLAGS_DEBUG}" MATCHES " /EH")
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHa")
    endif()
  endif()

  # Enable [[attribute]] syntax checking to prevent silent failure: "attribute is ignored in this syntactic position"
  add_extra_compiler_option("/w15240")

  if(NOT ENABLE_NOISY_WARNINGS)
    ocv_warnings_disable(CMAKE_CXX_FLAGS /wd4127) # conditional expression is constant
    ocv_warnings_disable(CMAKE_CXX_FLAGS /wd4251) # class 'std::XXX' needs to have dll-interface to be used by clients of YYY
    ocv_warnings_disable(CMAKE_CXX_FLAGS /wd4324) # 'struct_name' : structure was padded due to __declspec(align())
    ocv_warnings_disable(CMAKE_CXX_FLAGS /wd4275) # non dll-interface class 'std::exception' used as base for dll-interface class 'cv::Exception'
    ocv_warnings_disable(CMAKE_CXX_FLAGS /wd4512) # Assignment operator could not be generated
    ocv_warnings_disable(CMAKE_CXX_FLAGS /wd4589) # Constructor of abstract class 'cv::ORB' ignores initializer for virtual base class 'cv::Algorithm'
    ocv_warnings_disable(CMAKE_CXX_FLAGS /wd4819) # Symbols like delta or epsilon cannot be represented
  endif()

  if(CV_ICC AND NOT ENABLE_NOISY_WARNINGS)
    foreach(flags CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_RELEASE CMAKE_CXX_FLAGS_DEBUG CMAKE_C_FLAGS CMAKE_C_FLAGS_RELEASE CMAKE_C_FLAGS_DEBUG)
      string(REGEX REPLACE "( |^)/W[0-9]+( |$)" "\\1\\2" ${flags} "${${flags}}")
    endforeach()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Qwd673") # PCH warning
  endif()
endif()

if(APPLE AND NOT CMAKE_CROSSCOMPILING AND NOT DEFINED ENV{LDFLAGS} AND EXISTS "/usr/local/lib")
  link_directories("/usr/local/lib")
endif()

if(ENABLE_BUILD_HARDENING)
  include("${CMAKE_CURRENT_LIST_DIR}/OpenCVCompilerDefenses.cmake")
endif()

if(MSVC)
  include("${CMAKE_CURRENT_LIST_DIR}/OpenCVCRTLinkage.cmake")
  add_definitions(-D_VARIADIC_MAX=10)
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
  get_directory_property(__DIRECTORY_COMPILE_DEFINITIONS COMPILE_DEFINITIONS)
  if((NOT " ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE} ${OPENCV_EXTRA_CXX_FLAGS} ${OPENCV_EXTRA_FLAGS_RELEASE} ${__DIRECTORY_COMPILE_DEFINITIONS}" MATCHES "_WIN32_WINNT"
      AND NOT OPENCV_CMAKE_SKIP_MACRO_WIN32_WINNT)
      OR OPENCV_CMAKE_FORCE_MACRO_WIN32_WINNT
  )
    # https://docs.microsoft.com/en-us/cpp/porting/modifying-winver-and-win32-winnt
    # Target Windows 7 API
    set(OPENCV_CMAKE_MACRO_WIN32_WINNT "0x0601" CACHE STRING "Value of _WIN32_WINNT macro")
    add_definitions(-D_WIN32_WINNT=${OPENCV_CMAKE_MACRO_WIN32_WINNT})
  endif()
endif()

# Enable compiler options for OpenCV modules/apps/samples only (ignore 3rdparty)
macro(ocv_add_modules_compiler_options)
  if(MSVC AND NOT OPENCV_SKIP_MSVC_W4_OPTION)
    foreach(flags CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_RELEASE CMAKE_CXX_FLAGS_DEBUG)
      string(REPLACE "/W3" "/W4" ${flags} "${${flags}}")
    endforeach()
  endif()
  if(OPENCV_ENABLE_MEMORY_SANITIZER)
    add_definitions(-DOPENCV_ENABLE_MEMORY_SANITIZER=1)
  endif()
endmacro()

# adjust -Wl,-rpath-link
if(CMAKE_SKIP_RPATH)
  if((NOT CMAKE_CROSSCOMPILING OR OPENCV_ENABLE_LINKER_RPATH_LINK_ORIGIN) AND NOT OPENCV_SKIP_LINKER_RPATH_LINK_ORIGIN)
    if(DEFINED CMAKE_SHARED_LIBRARY_RPATH_ORIGIN_TOKEN)
      list(APPEND CMAKE_PLATFORM_RUNTIME_PATH "${CMAKE_SHARED_LIBRARY_RPATH_ORIGIN_TOKEN}")
    else()
      list(APPEND CMAKE_PLATFORM_RUNTIME_PATH "\$ORIGIN")
    endif()
  elseif(NOT OPENCV_SKIP_LINKER_RPATH_LINK_BINARY_LIB)
    list(APPEND CMAKE_PLATFORM_RUNTIME_PATH "${LIBRARY_OUTPUT_PATH}")
  endif()
endif()
if(OPENCV_EXTRA_RPATH_LINK_PATH)
  string(REPLACE ":" ";" OPENCV_EXTRA_RPATH_LINK_PATH_ "${OPENCV_EXTRA_RPATH_LINK_PATH}")
  list(APPEND CMAKE_PLATFORM_RUNTIME_PATH ${OPENCV_EXTRA_RPATH_LINK_PATH_})
  if(NOT CMAKE_EXECUTABLE_RPATH_LINK_CXX_FLAG)
    message(WARNING "OPENCV_EXTRA_RPATH_LINK_PATH may not work properly because CMAKE_EXECUTABLE_RPATH_LINK_CXX_FLAG is not defined (not supported)")
  endif()
endif()

# Control MSVC /MP flag
# Input variables: OPENCV_MSVC_PARALLEL (ON,1,2,3,...) + OPENCV_SKIP_MSVC_PARALLEL
# Details:
# - https://docs.microsoft.com/en-us/cpp/build/reference/mp-build-with-multiple-processes
# - https://docs.microsoft.com/en-us/cpp/build/reference/cl-environment-variables
# - https://gitlab.kitware.com/cmake/cmake/merge_requests/1718/diffs
if(CMAKE_GENERATOR MATCHES "Visual Studio" AND CMAKE_CXX_COMPILER_ID MATCHES "MSVC|Intel")
  ocv_check_environment_variables(OPENCV_SKIP_MSVC_PARALLEL)
  if(OPENCV_SKIP_MSVC_PARALLEL)
    # nothing
  elseif(" ${CMAKE_CXX_FLAGS}" MATCHES "/MP")
    # nothing, already defined in compiler flags
  elseif(DEFINED ENV{CL} AND " $ENV{CL}" MATCHES "/MP")
    # nothing, compiler will use CL environment variable
  elseif(DEFINED ENV{_CL_} AND " $ENV{_CL_}" MATCHES "/MP")
    # nothing, compiler will use _CL_ environment variable
  else()
    ocv_check_environment_variables(OPENCV_MSVC_PARALLEL)
    set(_mp_value "ON")
    if(DEFINED OPENCV_MSVC_PARALLEL)
      set(_mp_value "${OPENCV_MSVC_PARALLEL}")
    endif()
    set(OPENCV_MSVC_PARALLEL "${_mp_value}" CACHE STRING "Control MSVC /MP flag")
    if(_mp_value)
      if(_mp_value GREATER 0)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /MP${_mp_value}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP${_mp_value}")
      else()
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /MP")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
      endif()
    endif()
  endif()
endif()
