if(NOT PYTHON3_INCLUDE_DIRS OR NOT PYTHON3_NUMPY_INCLUDE_DIRS)
  ocv_module_disable(python3)
endif()

set(the_description "The python3 bindings")
set(MODULE_NAME python3)
set(MODULE_INSTALL_SUBDIR python3)

set(PYTHON PYTHON3)

include(../common.cmake)

unset(MODULE_NAME)
unset(MODULE_INSTALL_SUBDIR)
