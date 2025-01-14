###############################################################################
#
#  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
#
#  By downloading, copying, installing or using the software you agree to this license.
#  If you do not agree to this license, do not download, install,
#  copy or use the software.
#
#
#                           License Agreement
#                For Open Source Computer Vision Library
#
# Copyright (C) 2013, OpenCV Foundation, all rights reserved.
# Third party copyrights are property of their respective owners.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
#   * Redistribution's of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#
#   * Redistribution's in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#
#   * The name of the copyright holders may not be used to endorse or promote products
#     derived from this software without specific prior written permission.
#
# This software is provided by the copyright holders and contributors "as is" and
# any express or implied warranties, including, but not limited to, the implied
# warranties of merchantability and fitness for a particular purpose are disclaimed.
# In no event shall the Intel Corporation or contributors be liable for any direct,
# indirect, incidental, special, exemplary, or consequential damages
# (including, but not limited to, procurement of substitute goods or services;
# loss of use, data, or profits; or business interruption) however caused
# and on any theory of liability, whether in contract, strict liability,
# or tort (including negligence or otherwise) arising in any way out of
# the use of this software, even if advised of the possibility of such damage.
#

###############################################################################
# AUTHOR: Sajjad Taheri, University of California, Irvine. sajjadt[at]uci[dot]edu
#
#                             LICENSE AGREEMENT
# Copyright (c) 2015, 2015 The Regents of the University of California (Regents)
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
# 3. Neither the name of the University nor the
#    names of its contributors may be used to endorse or promote products
#    derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY COPYRIGHT HOLDERS AND CONTRIBUTORS ''AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
###############################################################################

from __future__ import print_function
import sys, re, os
from templates import *

if sys.version_info[0] >= 3:
    from io import StringIO
else:
    from cStringIO import StringIO

import json

func_table = {}

# Ignore these functions due to Embind limitations for now
ignore_list = ['locate',  #int&
               'minEnclosingCircle',  #float&
               'checkRange',
               'minMaxLoc',   #double*
               'floodFill', # special case, implemented in core_bindings.cpp
               'phaseCorrelate',
               'randShuffle',
               'calibrationMatrixValues', #double&
               'undistortPoints', # global redefinition
               'CamShift', #Rect&
               'meanShift' #Rect&
               ]

def makeWhiteList(module_list):
    wl = {}
    for m in module_list:
        for k in m.keys():
            if k in wl:
                wl[k] += m[k]
            else:
                wl[k] = m[k]
    return wl

def makeWhiteListJson(module_list):
    wl = {}
    for n, gen_dict in module_list.items():
        m = gen_dict["whitelist"]
        for k in m.keys():
            if k in wl:
                wl[k] += m[k]
            else:
                wl[k] = m[k]
    return wl

def makeNamespacePrefixOverride(module_list):
    wl = {}
    for n, gen_dict in module_list.items():
        if "namespace_prefix_override" in gen_dict:
            m = gen_dict["namespace_prefix_override"]
            for k in m.keys():
                if k in wl:
                    wl[k] += m[k]
                else:
                    wl[k] = m[k]
    return wl


white_list = None
namespace_prefix_override = None

# Features to be exported
export_enums = True
export_consts = True
with_wrapped_functions = True
with_default_params = True
with_vec_from_js_array = True

wrapper_namespace = "Wrappers"
type_dict = {
    'InputArray': 'const cv::Mat&',
    'OutputArray': 'cv::Mat&',
    'InputOutputArray': 'cv::Mat&',
    'InputArrayOfArrays': 'const std::vector<cv::Mat>&',
    'OutputArrayOfArrays': 'std::vector<cv::Mat>&',
    'string': 'std::string',
    'String': 'std::string',
    'const String&':'const std::string&'
}

def normalize_class_name(name):
    return re.sub(r"^cv\.", "", name).replace(".", "_")


class ClassProp(object):
    def __init__(self, decl):
        self.tp = decl[0].replace("*", "_ptr").strip()
        self.name = decl[1]
        self.readonly = True
        if "/RW" in decl[3]:
            self.readonly = False


class ClassInfo(object):
    def __init__(self, name, decl=None):
        self.cname = name.replace(".", "::")
        self.name = self.wname = normalize_class_name(name)

        self.ismap = False
        self.issimple = False
        self.isalgorithm = False
        self.methods = {}
        self.ext_constructors = {}
        self.props = []
        self.consts = {}
        customname = False
        self.jsfuncs = {}
        self.constructor_arg_num = set()

        self.has_smart_ptr = False

        if decl:
            self.bases = decl[1].split()[1:]
            if len(self.bases) > 1:
                self.bases = [self.bases[0].strip(",")]
                # return sys.exit(-1)
            if self.bases and self.bases[0].startswith("cv::"):
                self.bases[0] = self.bases[0][4:]
            if self.bases and self.bases[0] == "Algorithm":
                self.isalgorithm = True
            for m in decl[2]:
                if m.startswith("="):
                    self.wname = m[1:]
                    customname = True
                elif m == "/Map":
                    self.ismap = True
                elif m == "/Simple":
                    self.issimple = True
            self.props = [ClassProp(p) for p in decl[3]]

        if not customname and self.wname.startswith("Cv"):
            self.wname = self.wname[2:]


def handle_ptr(tp):
    if tp.startswith('Ptr_'):
        tp = 'Ptr<' + "::".join(tp.split('_')[1:]) + '>'
    return tp

def handle_vector(tp):
    if tp.startswith('vector_'):
        tp = handle_vector(tp[tp.find('_') + 1:])
        tp = 'std::vector<' + "::".join(tp.split('_')) + '>'
    return tp


class ArgInfo(object):
    def __init__(self, arg_tuple):
        self.tp = handle_ptr(arg_tuple[0]).strip()
        self.name = arg_tuple[1]
        self.defval = arg_tuple[2]
        self.isarray = False
        self.arraylen = 0
        self.arraycvt = None
        self.inputarg = True
        self.outputarg = False
        self.returnarg = False
        self.const = False
        self.reference = False
        for m in arg_tuple[3]:
            if m == "/O":
                self.inputarg = False
                self.outputarg = True
                self.returnarg = True
            elif m == "/IO":
                self.inputarg = True
                self.outputarg = True
                self.returnarg = True
            elif m.startswith("/A"):
                self.isarray = True
                self.arraylen = m[2:].strip()
            elif m.startswith("/CA"):
                self.isarray = True
                self.arraycvt = m[2:].strip()
            elif m == "/C":
                self.const = True
            elif m == "/Ref":
                self.reference = True
        if self.tp == "Mat" and (self.inputarg or self.outputarg):
            self.tp = "cv::Mat&"
            if self.inputarg and not self.outputarg:
                self.const = True
        if self.tp == "vector_Mat" and (self.inputarg or self.outputarg):
            self.tp = "std::vector<cv::Mat>&"
            if self.reference and not self.const:
                self.inputarg = False
                self.outputarg = True
            elif self.inputarg and not self.outputarg:
                self.const = True
        self.tp = handle_vector(self.tp).strip()
        if self.const:
            self.tp = "const " + self.tp
        if self.reference:
            self.tp = self.tp + "&"
        self.py_inputarg = False
        self.py_outputarg = False

class FuncVariant(object):
    def __init__(self, class_name, name, decl, is_constructor, is_class_method, is_const, is_virtual, is_pure_virtual, ref_return, const_return):
        self.class_name = class_name
        self.name = self.wname = name
        self.is_constructor = is_constructor
        self.is_class_method = is_class_method
        self.is_const = is_const
        self.is_virtual = is_virtual
        self.is_pure_virtual = is_pure_virtual
        self.refret = ref_return
        self.constret = const_return
        self.rettype = handle_vector(handle_ptr(decl[1]).strip()).strip()
        if self.rettype == "void":
            self.rettype = ""
        self.args = []
        self.array_counters = {}

        for a in decl[3]:
            ainfo = ArgInfo(a)
            if ainfo.isarray and not ainfo.arraycvt:
                c = ainfo.arraylen
                c_arrlist = self.array_counters.get(c, [])
                if c_arrlist:
                    c_arrlist.append(ainfo.name)
                else:
                    self.array_counters[c] = [ainfo.name]
            self.args.append(ainfo)


class FuncInfo(object):
    def __init__(self, class_name, name, cname, namespace, isconstructor):
        self.name_id = '_'.join([namespace] + ([class_name] if class_name else []) + [name])  # unique id for dict key

        self.class_name = class_name
        self.name = name
        self.cname = cname
        self.namespace = namespace
        self.variants = []
        self.is_constructor = isconstructor

    def add_variant(self, variant):
        self.variants.append(variant)


class Namespace(object):
    def __init__(self):
        self.funcs = {}
        self.enums = {}
        self.consts = {}


class JSWrapperGenerator(object):
    def __init__(self):

        self.bindings = []
        self.wrapper_funcs = []

        self.classes = {}  # FIXIT 'classes' should belong to 'namespaces'
        self.namespaces = {}
        self.enums = {}  # FIXIT 'enums' should belong to 'namespaces'

        self.parser = hdr_parser.CppHeaderParser()
        self.class_idx = 0

    def add_class(self, stype, name, decl):
        class_info = ClassInfo(name, decl)
        class_info.decl_idx = self.class_idx
        self.class_idx += 1

        if class_info.name in self.classes:
            print("Generator error: class %s (cpp_name=%s) already exists" \
                  % (class_info.name, class_info.cname))
            sys.exit(-1)
        self.classes[class_info.name] = class_info

    def resolve_class_inheritance(self):
        new_classes = {}
        for name, class_info in self.classes.items():

            if not hasattr(class_info, 'bases'):
                new_classes[name] = class_info
                continue # not class

            if class_info.bases:
                chunks = class_info.bases[0].split('::')
                base = '_'.join(chunks)
                while base not in self.classes and len(chunks) > 1:
                    del chunks[-2]
                    base = '_'.join(chunks)
                if base not in self.classes:
                    print("Generator error: unable to resolve base %s for %s"
                        % (class_info.bases[0], class_info.name))
                    sys.exit(-1)
                else:
                    class_info.bases[0] = "::".join(chunks)
                    class_info.isalgorithm |= self.classes[base].isalgorithm

            new_classes[name] = class_info

        self.classes = new_classes

    def split_decl_name(self, name):
        chunks = name.split('.')
        namespace = chunks[:-1]
        classes = []
        while namespace and '.'.join(namespace) not in self.parser.namespaces:
            classes.insert(0, namespace.pop())
        return namespace, classes, chunks[-1]

    def add_enum(self, decl):
        name = decl[0].rsplit(" ", 1)[1]
        namespace, classes, val = self.split_decl_name(name)
        namespace = '.'.join(namespace)
        ns = self.namespaces.setdefault(namespace, Namespace())
        if len(name) == 0: name = "<unnamed>"
        if name.endswith("<unnamed>"):
            i = 0
            while True:
                i += 1
                candidate_name = name.replace("<unnamed>", "unnamed_%u" % i)
                if candidate_name not in ns.enums:
                    name = candidate_name
                    break;
        cname = name.replace('.', '::')
        type_dict[normalize_class_name(name)] = cname
        if name in ns.enums:
            print("Generator warning: enum %s (cname=%s) already exists" \
                  % (name, cname))
            # sys.exit(-1)
        else:
            ns.enums[name] = []
        for item in decl[3]:
            ns.enums[name].append(item)

        const_decls = decl[3]

        for decl in const_decls:
            name = decl[0]
            self.add_const(name.replace("const ", "").strip(), decl)

    def add_const(self, name, decl):
        cname = name.replace('.','::')
        namespace, classes, name = self.split_decl_name(name)
        namespace = '.'.join(namespace)
        name = '_'.join(classes+[name])
        ns = self.namespaces.setdefault(namespace, Namespace())
        if name in ns.consts:
            print("Generator error: constant %s (cname=%s) already exists" \
                % (name, cname))
            sys.exit(-1)
        ns.consts[name] = cname

    def add_func(self, decl):
        namespace, classes, barename = self.split_decl_name(decl[0])
        cpp_name = "::".join(namespace + classes + [barename])
        name = barename
        class_name = ''
        bare_class_name = ''
        if classes:
            class_name = normalize_class_name('.'.join(namespace + classes))
            bare_class_name = classes[-1]
        namespace = '.'.join(namespace)

        is_constructor = name == bare_class_name
        is_class_method = False
        is_const_method = False
        is_virtual_method = False
        is_pure_virtual_method = False
        const_return = False
        ref_return = False

        for m in decl[2]:
            if m == "/S":
                is_class_method = True
            elif m == "/C":
                is_const_method = True
            elif m == "/V":
                is_virtual_method = True
            elif m == "/PV":
                is_pure_virtual_method = True
            elif m == "/Ref":
                ref_return = True
            elif m == "/CRet":
                const_return = True
            elif m.startswith("="):
                name = m[1:]

        if class_name:
            cpp_name = barename
            func_map = self.classes[class_name].methods
        else:
            func_map = self.namespaces.setdefault(namespace, Namespace()).funcs

        fi = FuncInfo(class_name, name, cpp_name, namespace, is_constructor)
        func = func_map.setdefault(fi.name_id, fi)

        variant = FuncVariant(class_name, name, decl, is_constructor, is_class_method, is_const_method,
                        is_virtual_method, is_pure_virtual_method, ref_return, const_return)
        func.add_variant(variant)

    def save(self, path, name, buf):
        f = open(path + "/" + name, "wt")
        f.write(buf.getvalue())
        f.close()

    def gen_function_binding_with_wrapper(self, func, ns_name, class_info):

        binding_text = None
        wrapper_func_text = None

        bindings = []
        wrappers = []

        for index, variant in enumerate(func.variants):

            factory = False
            if class_info and 'Ptr<' in variant.rettype:

                factory = True
                base_class_name = variant.rettype
                base_class_name = base_class_name.replace("Ptr<","").replace(">","").strip()
                if base_class_name in self.classes:
                    self.classes[base_class_name].has_smart_ptr = True
                else:
                    print(base_class_name, ' not found in classes for registering smart pointer using ', class_info.name, 'instead')
                    self.classes[class_info.name].has_smart_ptr = True

            def_args = []
            has_def_param = False

            # Return type
            ret_type = 'void' if variant.rettype.strip() == '' else variant.rettype
            if ret_type.startswith('Ptr'): #smart pointer
                ptr_type = ret_type.replace('Ptr<', '').replace('>', '')
                if ptr_type in type_dict:
                    ret_type = type_dict[ptr_type]
            for key in type_dict:
                if key in ret_type:
                    ret_type = re.sub(r"\b" + key + r"\b", type_dict[key], ret_type)
            arg_types = []
            unwrapped_arg_types = []
            for arg in variant.args:
                arg_type = None
                if arg.tp in type_dict:
                    arg_type = type_dict[arg.tp]
                else:
                    arg_type = arg.tp
                # Add default value
                if with_default_params and arg.defval != '':
                    def_args.append(arg.defval);
                arg_types.append(arg_type)
                unwrapped_arg_types.append(arg_type)

            # Function attribute
            func_attribs = ''
            if '*' in ''.join(arg_types):
                func_attribs += ', allow_raw_pointers()'

            if variant.is_pure_virtual:
                func_attribs += ', pure_virtual()'


            # Wrapper function
            if ns_name != None and ns_name != "cv":
                ns_parts = ns_name.split(".")
                if ns_parts[0] == "cv":
                    ns_parts = ns_parts[1:]
                ns_part = "_".join(ns_parts) + "_"
                ns_id = '_'.join(ns_parts)
                ns_prefix = namespace_prefix_override.get(ns_id, ns_id)
                if ns_prefix:
                    ns_prefix = ns_prefix + '_'
            else:
                ns_prefix = ''
            if class_info == None:
                js_func_name = ns_prefix + func.name
                wrap_func_name = js_func_name + "_wrapper"
            else:
                wrap_func_name = ns_prefix + func.class_name + "_" + func.name + "_wrapper"
                js_func_name = func.name

            # TODO: Name functions based wrap directives or based on arguments list
            if index > 0:
                wrap_func_name += str(index)
                js_func_name += str(index)

            c_func_name = 'Wrappers::' + wrap_func_name

            # Binding template-
            raw_arg_names = ['arg' + str(i + 1) for i in range(0, len(variant.args))]
            arg_names = []
            w_signature = []
            casted_arg_types = []
            for arg_type, arg_name in zip(arg_types, raw_arg_names):
                casted_arg_name = arg_name
                if with_vec_from_js_array:
                    # Only support const vector reference as input parameter
                    match = re.search(r'const std::vector<(.*)>&', arg_type)
                    if match:
                        type_in_vect = match.group(1)
                        if type_in_vect in ['int', 'float', 'double', 'char', 'uchar', 'String', 'std::string']:
                            casted_arg_name = 'emscripten::vecFromJSArray<' + type_in_vect + '>(' + arg_name + ')'
                            arg_type = re.sub(r'std::vector<(.*)>', 'emscripten::val', arg_type)
                w_signature.append(arg_type + ' ' + arg_name)
                arg_names.append(casted_arg_name)
                casted_arg_types.append(arg_type)

            arg_types = casted_arg_types

            # Argument list, signature
            arg_names_casted = [c if a == b else c + '.as<' + a + '>()' for a, b, c in
                                zip(unwrapped_arg_types, arg_types, arg_names)]

            # Add self object to the parameters
            if class_info and not  factory:
                arg_types = [class_info.cname + '&'] + arg_types
                w_signature = [class_info.cname + '& arg0 '] + w_signature

            for j in range(0, len(def_args) + 1):
                postfix = ''
                if j > 0:
                    postfix = '_' + str(j);

                ###################################
                # Wrapper
                if factory: # TODO or static
                    name = class_info.cname+'::' if variant.class_name else ""
                    cpp_call_text = static_class_call_template.substitute(scope=name,
                                                                   func=func.cname,
                                                                   args=', '.join(arg_names[:len(arg_names)-j]))
                elif class_info:
                    cpp_call_text = class_call_template.substitute(obj='arg0',
                                                                   func=func.cname,
                                                                   args=', '.join(arg_names[:len(arg_names)-j]))
                else:
                    cpp_call_text = call_template.substitute(func=func.cname,
                                                             args=', '.join(arg_names[:len(arg_names)-j]))


                wrapper_func_text = wrapper_function_template.substitute(ret_val=ret_type,
                                                                             func=wrap_func_name+postfix,
                                                                             signature=', '.join(w_signature[:len(w_signature)-j]),
                                                                             cpp_call=cpp_call_text,
                                                                             const='' if variant.is_const else '')

                ###################################
                # Binding
                if class_info:
                    if factory:
                        # print("Factory Function: ", c_func_name, len(variant.args) - j, class_info.name)
                        if variant.is_pure_virtual:
                            # FIXME: workaround for pure virtual in constructor
                            # e.g. DescriptorMatcher_clone_wrapper
                            continue
                        # consider the default parameter variants
                        args_num = len(variant.args) - j
                        if args_num in class_info.constructor_arg_num:
                            # FIXME: workaround for constructor overload with same args number
                            # e.g. DescriptorMatcher
                            continue
                        class_info.constructor_arg_num.add(args_num)
                        binding_text = ctr_template.substitute(const='const' if variant.is_const else '',
                                                           cpp_name=c_func_name+postfix,
                                                           ret=ret_type,
                                                           args=','.join(arg_types[:len(arg_types)-j]),
                                                           optional=func_attribs)
                    else:
                        binding_template = overload_class_static_function_template if variant.is_class_method else \
                            overload_class_function_template
                        binding_text = binding_template.substitute(js_name=js_func_name,
                                                           const='' if variant.is_const else '',
                                                           cpp_name=c_func_name+postfix,
                                                           ret=ret_type,
                                                           args=','.join(arg_types[:len(arg_types)-j]),
                                                           optional=func_attribs)
                else:
                    binding_text = overload_function_template.substitute(js_name=js_func_name,
                                                       cpp_name=c_func_name+postfix,
                                                       const='const' if variant.is_const else '',
                                                       ret=ret_type,
                                                       args=', '.join(arg_types[:len(arg_types)-j]),
                                                       optional=func_attribs)

                bindings.append(binding_text)
                wrappers.append(wrapper_func_text)

        return [bindings, wrappers]


    def gen_function_binding(self, func, class_info):

        if not class_info == None :
            func_name = class_info.cname+'::'+func.cname
        else :
            func_name = func.cname

        binding_text = None
        binding_text_list = []

        for index, variant in enumerate(func.variants):
            factory = False
            #TODO if variant.is_class_method and variant.rettype == ('Ptr<' + class_info.name + '>'):
            if (not class_info == None) and variant.rettype == ('Ptr<' + class_info.name + '>') or (func.name.startswith("create") and variant.rettype):
                factory = True
                base_class_name = variant.rettype
                base_class_name = base_class_name.replace("Ptr<","").replace(">","").strip()
                if base_class_name in self.classes:
                    self.classes[base_class_name].has_smart_ptr = True
                else:
                    print(base_class_name, ' not found in classes for registering smart pointer using ', class_info.name, 'instead')
                    self.classes[class_info.name].has_smart_ptr = True


            # Return type
            ret_type = 'void' if variant.rettype.strip() == '' else variant.rettype

            ret_type = ret_type.strip()
            if ret_type.startswith('Ptr'): #smart pointer
                ptr_type = ret_type.replace('Ptr<', '').replace('>', '')
                if ptr_type in type_dict:
                    ret_type = type_dict[ptr_type]
            for key in type_dict:
                if key in ret_type:
                    # Replace types. Instead of ret_type.replace we use regular
                    # expression to exclude false matches.
                    # See https://github.com/opencv/opencv/issues/15514
                    ret_type = re.sub(r"\b" + key + r"\b", type_dict[key], ret_type)
            if variant.constret and ret_type.startswith('const') == False:
                ret_type = 'const ' + ret_type
            if variant.refret and ret_type.endswith('&') == False:
                ret_type += '&'

            arg_types = []
            orig_arg_types = []
            def_args = []
            for arg in variant.args:
                if arg.tp in type_dict:
                    arg_type = type_dict[arg.tp]
                else:
                    arg_type = arg.tp

                #if arg.outputarg:
                #    arg_type += '&'
                orig_arg_types.append(arg_type)
                if with_default_params and arg.defval != '':
                    def_args.append(arg.defval)
                arg_types.append(orig_arg_types[-1])

            # Function attribute
            func_attribs = ''
            if '*' in ''.join(orig_arg_types):
                func_attribs += ', allow_raw_pointers()'

            if variant.is_pure_virtual:
                func_attribs += ', pure_virtual()'

            #TODO better naming
            #if variant.name in self.jsfunctions:
            #else
            js_func_name = variant.name


            c_func_name = func.cname if (factory and variant.is_class_method == False) else func_name


            ################################### Binding
            for j in range(0, len(def_args) + 1):
                postfix = ''
                if j > 0:
                    postfix = '_' + str(j);
                if factory:
                    binding_text = ctr_template.substitute(const='const' if variant.is_const else '',
                                                           cpp_name=c_func_name+postfix,
                                                           ret=ret_type,
                                                           args=','.join(arg_types[:len(arg_types)-j]),
                                                           optional=func_attribs)
                else:
                    binding_template = overload_class_static_function_template if variant.is_class_method else \
                            overload_function_template if class_info == None else overload_class_function_template
                    binding_text = binding_template.substitute(js_name=js_func_name,
                                                               const='const' if variant.is_const else '',
                                                               cpp_name=c_func_name+postfix,
                                                               ret=ret_type,
                                                               args=','.join(arg_types[:len(arg_types)-1]),
                                                               optional=func_attribs)

                binding_text_list.append(binding_text)

        return binding_text_list

    def print_decls(self, decls):
        """
        Prints the list of declarations, retrieived by the parse() method
        """
        for d in decls:
            print(d[0], d[1], ";".join(d[2]))
            for a in d[3]:
                print("   ", a[0], a[1], a[2], end="")
                if a[3]:
                    print("; ".join(a[3]))
                else:
                    print()

    def gen(self, dst_file, src_files, core_bindings):
        # step 1: scan the headers and extract classes, enums and functions
        headers = []
        for hdr in src_files:
            decls = self.parser.parse(hdr)
            # print(hdr);
            # self.print_decls(decls);
            if len(decls) == 0:
                continue
            headers.append(hdr[hdr.rindex('opencv2/'):])
            for decl in decls:
                name = decl[0]
                type = name[:name.find(" ")]
                if type == "struct" or type == "class":  # class/structure case
                    name = name[name.find(" ") + 1:].strip()
                    self.add_class(type, name, decl)
                elif name.startswith("enum"):  # enumerations
                    self.add_enum(decl)
                elif name.startswith("const"):
                    # constant
                    self.add_const(name.replace("const ", "").strip(), decl)
                else:  # class/global function
                    self.add_func(decl)

        self.resolve_class_inheritance()

        # step 2: generate bindings
        # Global functions
        for ns_name, ns in sorted(self.namespaces.items()):
            ns_parts = ns_name.split('.')
            if ns_parts[0] != 'cv':
                print('Ignore namespace: {}'.format(ns_name))
                continue
            else:
                ns_parts = ns_parts[1:]
            ns_id = '_'.join(ns_parts)
            ns_prefix = namespace_prefix_override.get(ns_id, ns_id)
            for name_id, func in sorted(ns.funcs.items()):
                name = func.name
                if ns_prefix:
                    name = ns_prefix + '_' + name
                if name in ignore_list:
                    continue
                if not name in white_list['']:
                    #print('Not in whitelist: "{}" from ns={}'.format(name, ns_name))
                    continue

                ext_cnst = False
                # Check if the method is an external constructor
                for variant in func.variants:
                    if "Ptr<" in variant.rettype:

                        # Register the smart pointer
                        base_class_name = variant.rettype
                        base_class_name = base_class_name.replace("Ptr<","").replace(">","").strip()
                        self.classes[base_class_name].has_smart_ptr = True

                        # Adds the external constructor
                        class_name = func.name.replace("create", "")
                        if not class_name in self.classes:
                            self.classes[base_class_name].methods[func.cname] = func
                        else:
                            self.classes[class_name].methods[func.cname] = func
                        ext_cnst = True
                if ext_cnst:
                    continue

                if with_wrapped_functions:
                    binding, wrapper = self.gen_function_binding_with_wrapper(func, ns_name, class_info=None)
                    self.bindings += binding
                    self.wrapper_funcs += wrapper
                else:
                    binding = self.gen_function_binding(func, class_info=None)
                    self.bindings+=binding

        # generate code for the classes and their methods
        for name, class_info in sorted(self.classes.items()):
            class_bindings = []
            if not name in white_list:
                #print('Not in whitelist: "{}" from ns={}'.format(name, ns_name))
                continue

            # Generate bindings for methods
            for method_name, method in sorted(class_info.methods.items()):
                if method.cname in ignore_list:
                    continue
                if not method.name in white_list[method.class_name]:
                    #print('Not in whitelist: "{}"'.format(method.name))
                    continue
                if method.is_constructor:
                    for variant in method.variants:
                        args = []
                        for arg in variant.args:
                            arg_type = type_dict[arg.tp] if arg.tp in type_dict else arg.tp
                            args.append(arg_type)
                        # print('Constructor: ', class_info.name, len(variant.args))
                        args_num = len(variant.args)
                        if args_num in class_info.constructor_arg_num:
                            continue
                        class_info.constructor_arg_num.add(args_num)
                        class_bindings.append(constructor_template.substitute(signature=', '.join(args)))
                else:
                    if with_wrapped_functions and (len(method.variants) > 1 or len(method.variants[0].args)>0 or "String" in method.variants[0].rettype):
                        binding, wrapper = self.gen_function_binding_with_wrapper(method, None, class_info=class_info)
                        self.wrapper_funcs = self.wrapper_funcs + wrapper
                        class_bindings = class_bindings + binding
                    else:
                        binding = self.gen_function_binding(method, class_info=class_info)
                        class_bindings = class_bindings + binding

            # Regiseter Smart pointer
            if class_info.has_smart_ptr:
                class_bindings.append(smart_ptr_reg_template.substitute(cname=class_info.cname, name=class_info.name))

            # Attach external constructors
            # for method_name, method in class_info.ext_constructors.items():
                # print("ext constructor", method_name)
            #if class_info.ext_constructors:



            # Generate bindings for properties
            for property in class_info.props:
                _class_property = class_property_enum_template if property.tp in type_dict else class_property_template
                class_bindings.append(_class_property.substitute(js_name=property.name, cpp_name='::'.join(
                    [class_info.cname, property.name])))

            dv = ''
            base = Template("""base<$base>""")

            assert len(class_info.bases) <= 1 , "multiple inheritance not supported"

            if len(class_info.bases) == 1:
                dv = "," + base.substitute(base=', '.join(class_info.bases))

            self.bindings.append(class_template.substitute(cpp_name=class_info.cname,
                                                           js_name=name,
                                                           class_templates=''.join(class_bindings),
                                                           derivation=dv))

        if export_enums:
            # step 4: generate bindings for enums
            # TODO anonymous enums are ignored for now.
            for ns_name, ns in sorted(self.namespaces.items()):
                if ns_name.split('.')[0] != 'cv':
                    continue
                for name, enum in sorted(ns.enums.items()):
                    if '.unnamed_' not in name:
                        name = name.replace("cv.", "")
                        enum_values = []
                        for enum_val in enum:
                            value = enum_val[0][enum_val[0].rfind(".")+1:]
                            enum_values.append(enum_item_template.substitute(val=value,
                                                                             cpp_val=name.replace('.', '::')+'::'+value))

                        self.bindings.append(enum_template.substitute(cpp_name=name.replace(".", "::"),
                                                                      js_name=name.replace(".", "_"),
                                                                      enum_items=''.join(enum_values)))
                    else:
                        print(name)
                        #TODO: represent anonymous enums with constants

        if export_consts:
            # step 5: generate bindings for consts
            for ns_name, ns in sorted(self.namespaces.items()):
                if ns_name.split('.')[0] != 'cv':
                    continue
                # TODO CALIB_FIX_FOCAL_LENGTH is defined both in cv:: and cv::fisheye
                prefix = 'FISHEYE_' if 'fisheye' in ns_name else ''
                for name, const in sorted(ns.consts.items()):
                    name = prefix + name
                    # print("Gen consts: ", name, const)
                    self.bindings.append(const_template.substitute(js_name=name, value=const))

        with open(core_bindings) as f:
            ret = f.read()

        header_includes = '\n'.join(['#include "{}"'.format(hdr) for hdr in headers])
        ret = ret.replace('@INCLUDES@', header_includes)

        defis = '\n'.join(self.wrapper_funcs)
        ret += wrapper_codes_template.substitute(ns=wrapper_namespace, defs=defis)
        ret += emscripten_binding_template.substitute(binding_name='testBinding', bindings=''.join(self.bindings))


        # print(ret)
        text_file = open(dst_file, "w")
        text_file.write(ret)
        text_file.close()


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage:\n", \
            os.path.basename(sys.argv[0]), \
            "<full path to hdr_parser.py> <bindings.cpp> <headers.txt> <core_bindings.cpp> <whitelist.json or opencv_js.config.py>")
        print("Current args are: ", ", ".join(["'"+a+"'" for a in sys.argv]))
        exit(1)

    dstdir = "."
    hdr_parser_path = os.path.abspath(sys.argv[1])
    if hdr_parser_path.endswith(".py"):
        hdr_parser_path = os.path.dirname(hdr_parser_path)
    sys.path.append(hdr_parser_path)
    import hdr_parser

    bindingsCpp = sys.argv[2]
    headers = open(sys.argv[3], 'r').read().split(';')
    coreBindings = sys.argv[4]
    whiteListFile = sys.argv[5]

    if whiteListFile.endswith(".json") or whiteListFile.endswith(".JSON"):
        with open(whiteListFile) as f:
            gen_dict = json.load(f)
        f.close()
        white_list = makeWhiteListJson(gen_dict)
        namespace_prefix_override = makeNamespacePrefixOverride(gen_dict)
    elif whiteListFile.endswith(".py") or whiteListFile.endswith(".PY"):
        exec(open(whiteListFile).read())
        assert(white_list)
        namespace_prefix_override = {
            'dnn' : '',
            'aruco' : '',
        }
    else:
        print("Unexpected format of OpenCV config file", whiteListFile)
        exit(1)

    generator = JSWrapperGenerator()
    generator.gen(bindingsCpp, headers, coreBindings)
