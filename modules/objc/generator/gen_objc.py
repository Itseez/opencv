#!/usr/bin/env python3

from __future__ import print_function, unicode_literals
import sys, re, os.path, errno, fnmatch
import json
import logging
import codecs
import io
from shutil import copyfile
from pprint import pformat
from string import Template

if sys.version_info >= (3, 8): # Python 3.8+
    from shutil import copytree
    def copy_tree(src, dst):
        copytree(src, dst, dirs_exist_ok=True)
else:
    from distutils.dir_util import copy_tree

try:
    from io import StringIO # Python 3
except:
    from io import BytesIO as StringIO

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# list of modules
config = None
ROOT_DIR = None

total_files = 0
updated_files = 0

module_imports = []

# list of namespaces, which should be skipped by wrapper generator
# the list is loaded from misc/objc/gen_dict.json defined for the module only
namespace_ignore_list = []

# list of class names, which should be skipped by wrapper generator
# the list is loaded from misc/objc/gen_dict.json defined for the module and its dependencies
class_ignore_list = []


# list of enum names, which should be skipped by wrapper generator
enum_ignore_list = []

# list of constant names, which should be skipped by wrapper generator
# ignored constants can be defined using regular expressions
const_ignore_list = []

# list of private constants
const_private_list = []

# { Module : { public : [[name, val],...], private : [[]...] } }
missing_consts = {}

type_dict = {
    ""        : {"objc_type" : ""}, # c-tor ret_type
    "void"    : {"objc_type" : "void", "is_primitive" : True, "swift_type": "Void"},
    "bool"    : {"objc_type" : "BOOL", "is_primitive" : True, "to_cpp": "(bool)%(n)s", "swift_type": "Bool"},
    "char"    : {"objc_type" : "char", "is_primitive" : True, "swift_type": "Int8"},
    "int"     : {"objc_type" : "int", "is_primitive" : True, "out_type" : "int*", "out_type_ptr": "%(n)s", "out_type_ref": "*(int*)(%(n)s)", "swift_type": "Int32"},
    "long"    : {"objc_type" : "long", "is_primitive" : True, "swift_type": "Int"},
    "float"   : {"objc_type" : "float", "is_primitive" : True, "out_type" : "float*", "out_type_ptr": "%(n)s", "out_type_ref": "*(float*)(%(n)s)", "swift_type": "Float"},
    "double"  : {"objc_type" : "double", "is_primitive" : True, "out_type" : "double*", "out_type_ptr": "%(n)s", "out_type_ref": "*(double*)(%(n)s)", "swift_type": "Double"},
    "size_t"  : {"objc_type" : "size_t", "is_primitive" : True},
    "int64"   : {"objc_type" : "long", "is_primitive" : True, "swift_type": "Int"},
    "string"  : {"objc_type" : "NSString*", "is_primitive" : True, "from_cpp": "[NSString stringWithUTF8String:%(n)s.c_str()]", "cast_to": "std::string", "swift_type": "String"}
}

# Defines a rule to add extra prefixes for names from specific namespaces.
# In example, cv::fisheye::stereoRectify from namespace fisheye is wrapped as fisheye_stereoRectify
namespaces_dict = {}

# { module: { class | "*" : [ header ]} }
AdditionalImports = {}

# { class : { func : {declaration, implementation} } }
ManualFuncs = {}

# { class : { func : { arg_name : {"ctype" : ctype, "attrib" : [attrib]} } } }
func_arg_fix = {}

# { class : { func : { prolog : "", epilog : "" } } }
header_fix = {}

# { class : { enum: fixed_enum } }
enum_fix = {}

# { class : { enum: { const: fixed_const} } }
const_fix = {}

# { (class, func) : objc_signature }
method_dict = {
    ("Mat", "convertTo") : "-convertTo:rtype:alpha:beta:",
    ("Mat", "setTo") : "-setToScalar:mask:",
    ("Mat", "zeros") : "+zeros:cols:type:",
    ("Mat", "ones") : "+ones:cols:type:",
    ("Mat", "dot") : "-dot:"
}

modules = []


class SkipSymbolException(Exception):
    def __init__(self, text):
        self.t = text
    def __str__(self):
        return self.t


def read_contents(fname):
    with open(fname, 'r') as f:
        data = f.read()
    return data

def mkdir_p(path):
    ''' mkdir -p '''
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def header_import(hdr):
    """ converts absolute header path to import parameter """
    pos = hdr.find('/include/')
    hdr = hdr[pos+9 if pos >= 0 else 0:]
    #pos = hdr.find('opencv2/')
    #hdr = hdr[pos+8 if pos >= 0 else 0:]
    return hdr

def make_objcname(m):
    return "Cv"+m if (m[0] in "0123456789") else m

def make_objcmodule(m):
    return "cv"+m if (m[0] in "0123456789") else m

T_OBJC_CLASS_HEADER = read_contents(os.path.join(SCRIPT_DIR, 'templates/objc_class_header.template'))
T_OBJC_CLASS_BODY = read_contents(os.path.join(SCRIPT_DIR, 'templates/objc_class_body.template'))
T_OBJC_MODULE_HEADER = read_contents(os.path.join(SCRIPT_DIR, 'templates/objc_module_header.template'))
T_OBJC_MODULE_BODY = read_contents(os.path.join(SCRIPT_DIR, 'templates/objc_module_body.template'))

class GeneralInfo():
    def __init__(self, type, decl, namespaces):
        self.symbol_id, self.namespace, self.classpath, self.classname, self.name = self.parseName(decl[0], namespaces)

        for ns_ignore in namespace_ignore_list:
            if self.symbol_id.startswith(ns_ignore + '.'):
                raise SkipSymbolException('ignored namespace ({}): {}'.format(ns_ignore, self.symbol_id))

        # parse doxygen comments
        self.params={}

        self.deprecated = False
        if type == "class":
            docstring = "// C++: class " + self.name + "\n"
        else:
            docstring=""

        if len(decl)>5 and decl[5]:
            doc = decl[5]

            if re.search("(@|\\\\)deprecated", doc):
                self.deprecated = True

            docstring += sanitize_documentation_string(doc, type)
        elif type == "class":
            docstring += "/**\n * The " + self.name + " module\n */\n"

        self.docstring = docstring

    def parseName(self, name, namespaces):
        '''
        input: full name and available namespaces
        returns: (namespace, classpath, classname, name)
        '''
        name = name[name.find(" ")+1:].strip() # remove struct/class/const prefix
        spaceName = ""
        localName = name # <classes>.<name>
        for namespace in sorted(namespaces, key=len, reverse=True):
            if name.startswith(namespace + "."):
                spaceName = namespace
                localName = name.replace(namespace + ".", "")
                break
        pieces = localName.split(".")
        if len(pieces) > 2: # <class>.<class>.<class>.<name>
            return name, spaceName, ".".join(pieces[:-1]), pieces[-2], pieces[-1]
        elif len(pieces) == 2: # <class>.<name>
            return name, spaceName, pieces[0], pieces[0], pieces[1]
        elif len(pieces) == 1: # <name>
            return name, spaceName, "", "", pieces[0]
        else:
            return name, spaceName, "", "" # error?!

    def fullName(self, isCPP=False):
        result = ".".join([self.fullClass(), self.name])
        return result if not isCPP else get_cname(result)

    def fullClass(self, isCPP=False):
        result = ".".join([f for f in [self.namespace] + self.classpath.split(".") if len(f)>0])
        return result if not isCPP else get_cname(result)

class ConstInfo(GeneralInfo):
    def __init__(self, decl, addedManually=False, namespaces=[], enumType=None):
        GeneralInfo.__init__(self, "const", decl, namespaces)
        self.cname = get_cname(self.name)
        self.swift_name = None
        self.value = decl[1]
        self.enumType = enumType
        self.addedManually = addedManually
        if self.namespace in namespaces_dict:
            self.name = '%s_%s' % (namespaces_dict[self.namespace], self.name)

    def __repr__(self):
        return Template("CONST $name=$value$manual").substitute(name=self.name,
                                                                 value=self.value,
                                                                 manual="(manual)" if self.addedManually else "")

    def isIgnored(self):
        for c in const_ignore_list:
            if re.match(c, self.name):
                return True
        return False

def normalize_field_name(name):
    return name.replace(".","_").replace("[","").replace("]","").replace("_getNativeObjAddr()","_nativeObj")

def normalize_class_name(name):
    return re.sub(r"^cv\.", "", name).replace(".", "_")

def get_cname(name):
    return name.replace(".", "::")

def cast_from(t):
    if t in type_dict and "cast_from" in type_dict[t]:
        return type_dict[t]["cast_from"]
    return t

def cast_to(t):
    if t in type_dict and "cast_to" in type_dict[t]:
        return type_dict[t]["cast_to"]
    return t

def gen_class_doc(docstring, module, members, enums):
    lines = docstring.splitlines()
    lines.insert(len(lines)-1, " *")
    if len(members) > 0:
        lines.insert(len(lines)-1, " * Member classes: " + ", ".join([("`" + m + "`") for m in members]))
        lines.insert(len(lines)-1, " *")
    else:
        lines.insert(len(lines)-1, " * Member of `" + module + "`")
    if len(enums) > 0:
        lines.insert(len(lines)-1, " * Member enums: " + ", ".join([("`" + m + "`") for m in enums]))

    return "\n".join(lines)

class ClassPropInfo():
    def __init__(self, decl): # [f_ctype, f_name, '', '/RW']
        self.ctype = decl[0]
        self.name = decl[1]
        self.rw = "/RW" in decl[3]

    def __repr__(self):
        return Template("PROP $ctype $name").substitute(ctype=self.ctype, name=self.name)

class ClassInfo(GeneralInfo):
    def __init__(self, decl, namespaces=[]): # [ 'class/struct cname', ': base', [modlist] ]
        GeneralInfo.__init__(self, "class", decl, namespaces)
        self.cname = self.name if not self.classname else self.classname + "_" + self.name
        self.real_cname = self.name if not self.classname else self.classname + "::" + self.name
        self.methods = []
        self.methods_suffixes = {}
        self.consts = [] # using a list to save the occurrence order
        self.private_consts = []
        self.imports = set()
        self.props= []
        self.objc_name = self.name if not self.classname else self.classname + self.name
        self.smart = None # True if class stores Ptr<T>* instead of T* in nativeObj field
        self.additionalImports = None # additional import files
        self.enum_declarations = None # Objective-C enum declarations stream
        self.method_declarations = None # Objective-C method declarations stream
        self.method_implementations = None # Objective-C method implementations stream
        self.objc_header_template = None # Objective-C header code
        self.objc_body_template = None # Objective-C body code
        for m in decl[2]:
            if m.startswith("="):
                self.objc_name = m[1:]
        self.base = ''
        self.is_base_class = True
        self.native_ptr_name = "nativePtr"
        self.member_classes = [] # Only relevant for modules
        self.member_enums = [] # Only relevant for modules
        if decl[1]:
            self.base = re.sub(r"^.*:", "", decl[1].split(",")[0]).strip()
            if self.base:
                self.is_base_class = False
                self.native_ptr_name = "nativePtr" + self.objc_name

    def __repr__(self):
        return Template("CLASS $namespace::$classpath.$name : $base").substitute(**self.__dict__)

    def getImports(self, module):
        return ["#import \"%s.h\"" % make_objcname(c) for c in sorted([m for m in [type_dict[m]["import_module"] if m in type_dict and "import_module" in type_dict[m] else m for m in self.imports] if m != self.name])]

    def isEnum(self, c):
        return c in type_dict and type_dict[c].get("is_enum", False)

    def getForwardDeclarations(self, module):
        enum_decl = [x for x in self.imports if self.isEnum(x) and type_dict[x]["import_module"] != module]
        enum_imports = sorted(list(set([type_dict[m]["import_module"] for m in enum_decl])))
        class_decl = [x for x in self.imports if not self.isEnum(x)]
        return ["#import \"%s.h\"" % make_objcname(c) for c in enum_imports] + [""] + ["@class %s;" % c for c in sorted(class_decl)]

    def addImports(self, ctype, is_out_type):
        if ctype == self.cname:
            return
        if ctype in type_dict:
            objc_import = None
            if "v_type" in type_dict[ctype]:
                objc_import = type_dict[type_dict[ctype]["v_type"]]["objc_type"]
            elif "v_v_type" in type_dict[ctype]:
                objc_import = type_dict[type_dict[ctype]["v_v_type"]]["objc_type"]
            elif not type_dict[ctype].get("is_primitive", False):
                objc_import = type_dict[ctype]["objc_type"]
            if objc_import is not None and objc_import not in ["NSNumber*", "NSString*"] and not (objc_import in type_dict and type_dict[objc_import].get("is_primitive", False)):
                objc_import = objc_import[:-1] if objc_import[-1] == "*" else objc_import   # remove trailing "*"
                if objc_import != self.cname:
                    self.imports.add(objc_import)   # remove trailing "*"

    def getAllMethods(self):
        result = []
        result += [fi for fi in self.methods if fi.isconstructor]
        result += [fi for fi in self.methods if not fi.isconstructor]
        return result

    def addMethod(self, fi):
        self.methods.append(fi)

    def getConst(self, name):
        for cand in self.consts + self.private_consts:
            if cand.name == name:
                return cand
        return None

    def addConst(self, constinfo):
        # choose right list (public or private)
        consts = self.consts
        for c in const_private_list:
            if re.match(c, constinfo.name):
                consts = self.private_consts
                break
        consts.append(constinfo)

    def initCodeStreams(self, Module):
        self.additionalImports = StringIO()
        self.enum_declarations = StringIO()
        self.method_declarations = StringIO()
        self.method_implementations = StringIO()
        if self.base:
            self.objc_header_template = T_OBJC_CLASS_HEADER
            self.objc_body_template = T_OBJC_CLASS_BODY
        else:
            self.base = "NSObject"
            if self.name != Module:
                self.objc_header_template = T_OBJC_CLASS_HEADER
                self.objc_body_template = T_OBJC_CLASS_BODY
            else:
                self.objc_header_template = T_OBJC_MODULE_HEADER
                self.objc_body_template = T_OBJC_MODULE_BODY
        # misc handling
        if self.name == Module:
          for i in module_imports or []:
              self.imports.add(i)

    def cleanupCodeStreams(self):
        self.additionalImports.close()
        self.enum_declarations.close()
        self.method_declarations.close()
        self.method_implementations.close()

    def generateObjcHeaderCode(self, m, M, objcM):
        return Template(self.objc_header_template + "\n\n").substitute(
                            module = M,
                            additionalImports = self.additionalImports.getvalue(),
                            importBaseClass = '#import "' + make_objcname(self.base) + '.h"' if not self.is_base_class else "",
                            forwardDeclarations = "\n".join([_f for _f in self.getForwardDeclarations(objcM) if _f]),
                            enumDeclarations = self.enum_declarations.getvalue(),
                            nativePointerHandling = Template(
"""
#ifdef __cplusplus
@property(readonly)cv::Ptr<$cName> $native_ptr_name;
#endif

#ifdef __cplusplus
- (instancetype)initWithNativePtr:(cv::Ptr<$cName>)nativePtr;
+ (instancetype)fromNative:(cv::Ptr<$cName>)nativePtr;
#endif
"""
                            ).substitute(
                                cName = self.fullName(isCPP=True),
                                native_ptr_name = self.native_ptr_name
                            ),
                            manualMethodDeclations = "",
                            methodDeclarations = self.method_declarations.getvalue(),
                            name = self.name,
                            objcName = make_objcname(self.objc_name),
                            cName = self.cname,
                            imports = "\n".join(self.getImports(M)),
                            docs = gen_class_doc(self.docstring, M, self.member_classes, self.member_enums),
                            base = self.base)

    def generateObjcBodyCode(self, m, M):
        return Template(self.objc_body_template + "\n\n").substitute(
                            module = M,
                            objcname = make_objcname(M),
                            nativePointerHandling=Template(
"""
- (instancetype)initWithNativePtr:(cv::Ptr<$cName>)nativePtr {
    self = [super $init_call];
    if (self) {
        _$native_ptr_name = nativePtr;
    }
    return self;
}

+ (instancetype)fromNative:(cv::Ptr<$cName>)nativePtr {
    return [[$objcName alloc] initWithNativePtr:nativePtr];
}
"""
                            ).substitute(
                                cName = self.fullName(isCPP=True),
                                objcName = make_objcname(self.objc_name),
                                native_ptr_name = self.native_ptr_name,
                                init_call = "init" if self.is_base_class else "initWithNativePtr:nativePtr"
                            ),
                            manualMethodDeclations = "",
                            methodImplementations = self.method_implementations.getvalue(),
                            name = self.name,
                            objcName = make_objcname(self.objc_name),
                            cName = self.cname,
                            imports = "\n".join(self.getImports(M)),
                            docs = gen_class_doc(self.docstring, M, self.member_classes, self.member_enums),
                            base = self.base)

class ArgInfo():
    def __init__(self, arg_tuple): # [ ctype, name, def val, [mod], argno ]
        self.pointer = False
        ctype = arg_tuple[0]
        if ctype.endswith("*"):
            ctype = ctype[:-1]
            self.pointer = True
        self.ctype = ctype
        self.name = arg_tuple[1]
        self.defval = arg_tuple[2]
        self.out = ""
        if "/O" in arg_tuple[3]:
            self.out = "O"
        if "/IO" in arg_tuple[3]:
            self.out = "IO"

    def __repr__(self):
        return Template("ARG $ctype$p $name=$defval").substitute(ctype=self.ctype,
                                                                  p=" *" if self.pointer else "",
                                                                  name=self.name,
                                                                  defval=self.defval)

class FuncInfo(GeneralInfo):
    def __init__(self, decl, module, namespaces=[]): # [ funcname, return_ctype, [modifiers], [args] ]
        GeneralInfo.__init__(self, "func", decl, namespaces)
        self.cname = get_cname(decl[0])
        nested_type = self.classpath.find(".") != -1
        self.objc_name = self.name if not nested_type else self.classpath.replace(".", "")
        self.classname = self.classname if not nested_type else self.classpath.replace(".", "_")
        self.swift_name = self.name
        self.cv_name = self.fullName(isCPP=True)
        self.isconstructor = self.name == self.classname
        if "[" in self.name:
            self.objc_name = "getelem"
        if self.namespace in namespaces_dict:
            self.objc_name = '%s_%s' % (namespaces_dict[self.namespace], self.objc_name)
            self.swift_name = '%s_%s' % (namespaces_dict[self.namespace], self.swift_name)
        for m in decl[2]:
            if m.startswith("="):
                self.objc_name = m[1:]
        self.static = ["","static"][ "/S" in decl[2] ]
        self.ctype = re.sub(r"^CvTermCriteria", "TermCriteria", decl[1] or "")
        self.args = []
        func_fix_map = func_arg_fix.get(self.classname or module, {}).get(self.objc_name, {})
        header_fixes = header_fix.get(self.classname or module, {}).get(self.objc_name, {})
        self.prolog = header_fixes.get('prolog', None)
        self.epilog = header_fixes.get('epilog', None)
        for a in decl[3]:
            arg = a[:]
            arg_fix_map = func_fix_map.get(arg[1], {})
            arg[0] = arg_fix_map.get('ctype',  arg[0]) #fixing arg type
            arg[2] = arg_fix_map.get('defval', arg[2]) #fixing arg defval
            arg[3] = arg_fix_map.get('attrib', arg[3]) #fixing arg attrib
            self.args.append(ArgInfo(arg))

        if type_complete(self.args, self.ctype):
            func_fix_map = func_arg_fix.get(self.classname or module, {}).get(self.signature(self.args), {})
            name_fix_map = func_fix_map.get(self.name, {})
            self.objc_name = name_fix_map.get('name', self.objc_name)
            self.swift_name = name_fix_map.get('swift_name', self.swift_name)
            for arg in self.args:
                arg_fix_map = func_fix_map.get(arg.name, {})
                arg.ctype = arg_fix_map.get('ctype', arg.ctype) #fixing arg type
                arg.defval = arg_fix_map.get('defval', arg.defval) #fixing arg type
                arg.name = arg_fix_map.get('name', arg.name) #fixing arg name

    def __repr__(self):
        return Template("FUNC <$ctype $namespace.$classpath.$name $args>").substitute(**self.__dict__)

    def __lt__(self, other):
        return self.__repr__() < other.__repr__()

    def signature(self, args):
        objc_args = build_objc_args(args)
        return "(" + type_dict[self.ctype]["objc_type"] + ")" + self.objc_name + " ".join(objc_args)

def type_complete(args, ctype):
    for a in args:
        if a.ctype not in type_dict:
            if not a.defval and a.ctype.endswith("*"):
                a.defval = 0
            if a.defval:
                a.ctype = ''
                continue
            return False
    if ctype not in type_dict:
        return False
    return True

def build_objc_args(args):
    objc_args = []
    for a in args:
        if a.ctype not in type_dict:
            if not a.defval and a.ctype.endswith("*"):
                a.defval = 0
            if a.defval:
                a.ctype = ''
                continue
        if not a.ctype:  # hidden
            continue
        objc_type = type_dict[a.ctype]["objc_type"]
        if "v_type" in type_dict[a.ctype]:
            if "O" in a.out:
                objc_type = "NSMutableArray<" + objc_type + ">*"
            else:
                objc_type = "NSArray<" + objc_type + ">*"
        elif "v_v_type" in type_dict[a.ctype]:
            if "O" in a.out:
                objc_type = "NSMutableArray<NSMutableArray<" + objc_type + ">*>*"
            else:
                objc_type = "NSArray<NSArray<" + objc_type + ">*>*"

        if a.out and type_dict[a.ctype].get("out_type", ""):
            objc_type = type_dict[a.ctype]["out_type"]
        objc_args.append((a.name if len(objc_args) > 0 else '') + ':(' + objc_type + ')' + a.name)
    return objc_args

def build_objc_method_name(args):
    objc_method_name = ""
    for a in args[1:]:
        if a.ctype not in type_dict:
            if not a.defval and a.ctype.endswith("*"):
                a.defval = 0
            if a.defval:
                a.ctype = ''
                continue
        if not a.ctype:  # hidden
            continue
        objc_method_name += a.name + ":"
    return objc_method_name

def get_swift_type(ctype):
    has_swift_type = "swift_type" in type_dict[ctype]
    swift_type = type_dict[ctype]["swift_type"] if has_swift_type else type_dict[ctype]["objc_type"]
    if swift_type[-1:] == "*":
        swift_type = swift_type[:-1]
    if not has_swift_type:
        if "v_type" in type_dict[ctype]:
            swift_type = "[" + swift_type + "]"
        elif "v_v_type" in type_dict[ctype]:
            swift_type = "[[" + swift_type + "]]"
    return swift_type

def build_swift_extension_decl(name, args, constructor, static, ret_type):
    extension_decl = "@nonobjc " + ("class " if static else "") + (("func " + name) if not constructor else "convenience init") + "("
    swift_args = []
    for a in args:
        if a.ctype not in type_dict:
            if not a.defval and a.ctype.endswith("*"):
                a.defval = 0
            if a.defval:
                a.ctype = ''
                continue
        if not a.ctype:  # hidden
            continue
        swift_type = get_swift_type(a.ctype)

        if "O" in a.out:
            if type_dict[a.ctype].get("primitive_type", False):
                swift_type = "UnsafeMutablePointer<" + swift_type + ">"
            elif "v_type" in type_dict[a.ctype] or "v_v_type" in type_dict[a.ctype] or type_dict[a.ctype].get("primitive_vector", False) or type_dict[a.ctype].get("primitive_vector_vector", False):
                swift_type = "inout " + swift_type

        swift_args.append(a.name + ': ' + swift_type)

    extension_decl += ", ".join(swift_args) + ")"
    if ret_type:
        extension_decl += " -> " + get_swift_type(ret_type)
    return extension_decl

def extension_arg(a):
    return a.ctype in type_dict and (type_dict[a.ctype].get("primitive_vector", False) or type_dict[a.ctype].get("primitive_vector_vector", False) or (("v_type" in type_dict[a.ctype] or "v_v_type" in type_dict[a.ctype]) and "O" in a.out))

def extension_tmp_arg(a):
    if a.ctype in type_dict:
        if type_dict[a.ctype].get("primitive_vector", False) or type_dict[a.ctype].get("primitive_vector_vector", False):
            return a.name + "Vector"
        elif ("v_type" in type_dict[a.ctype] or "v_v_type" in type_dict[a.ctype]) and "O" in a.out:
            return a.name + "Array"
    return a.name

def make_swift_extension(args):
    for a in args:
        if extension_arg(a):
            return True
    return False

def build_swift_signature(args):
    swift_signature = ""
    for a in args:
        if a.ctype not in type_dict:
            if not a.defval and a.ctype.endswith("*"):
                a.defval = 0
            if a.defval:
                a.ctype = ''
                continue
        if not a.ctype:  # hidden
            continue
        swift_signature += a.name + ":"
    return swift_signature

def build_unrefined_call(name, args, constructor, static, classname, has_ret):
    swift_refine_call = ("let ret = " if has_ret and not constructor else "") + ((make_objcname(classname) + ".") if static else "") + (name if not constructor else "self.init")
    call_args = []
    for a in args:
        if a.ctype not in type_dict:
            if not a.defval and a.ctype.endswith("*"):
                a.defval = 0
            if a.defval:
                a.ctype = ''
                continue
        if not a.ctype:  # hidden
            continue
        call_args.append(a.name + ": " + extension_tmp_arg(a))
    swift_refine_call += "(" + ", ".join(call_args) + ")"
    return swift_refine_call

def build_swift_logues(args):
    prologue = []
    epilogue = []
    for a in args:
        if a.ctype not in type_dict:
            if not a.defval and a.ctype.endswith("*"):
                a.defval = 0
            if a.defval:
                a.ctype = ''
                continue
        if not a.ctype:  # hidden
            continue
        if a.ctype in type_dict:
            if type_dict[a.ctype].get("primitive_vector", False):
                prologue.append("let " + extension_tmp_arg(a) + " = " + type_dict[a.ctype]["objc_type"][:-1] + "(" + a.name + ")")
                if "O" in a.out:
                    unsigned = type_dict[a.ctype].get("unsigned", False)
                    array_prop = "array" if not unsigned else "unsignedArray"
                    epilogue.append(a.name + ".removeAll()")
                    epilogue.append(a.name + ".append(contentsOf: " +  extension_tmp_arg(a) + "." + array_prop + ")")
            elif type_dict[a.ctype].get("primitive_vector_vector", False):
                if not "O" in a.out:
                    prologue.append("let " + extension_tmp_arg(a) + " = " + a.name + ".map {" + type_dict[a.ctype]["objc_type"][:-1] + "($0) }")
                else:
                    prologue.append("let " + extension_tmp_arg(a) + " = NSMutableArray(array: " + a.name + ".map {" + type_dict[a.ctype]["objc_type"][:-1] + "($0) })")
                    epilogue.append(a.name + ".removeAll()")
                    epilogue.append(a.name + ".append(contentsOf: " + extension_tmp_arg(a) + ".map { ($.0 as! " + type_dict[a.ctype]["objc_type"][:-1] + ").array  })")
            elif ("v_type" in type_dict[a.ctype] or "v_v_type" in type_dict[a.ctype]) and "O" in a.out:
                prologue.append("let " +  extension_tmp_arg(a) + " = NSMutableArray(array: " + a.name + ")")
                epilogue.append(a.name + ".removeAll()")
                epilogue.append(a.name + ".append(contentsOf: " +  extension_tmp_arg(a) + " as! " + get_swift_type(a.ctype) + ")")
    return prologue, epilogue

def add_method_to_dict(class_name, fi):
    static = fi.static if fi.classname else True
    if (class_name, fi.objc_name) not in method_dict:
        objc_method_name = ("+" if static else "-") + fi.objc_name + ":" + build_objc_method_name(fi.args)
        method_dict[(class_name, fi.objc_name)] = objc_method_name

def see_lookup(objc_class, see):
    semi_colon = see.find("::")
    see_class = see[:semi_colon] if semi_colon > 0 else objc_class
    see_method = see[(semi_colon + 2):] if semi_colon != -1 else see
    if (see_class, see_method) in method_dict:
        method = method_dict[(see_class, see_method)]
        if see_class == objc_class:
            return method
        else:
            return ("-" if method[0] == "-" else "") + "[" + see_class + " " + method[1:] + "]"
    else:
        return see


class ObjectiveCWrapperGenerator(object):
    def __init__(self):
        self.header_files = []
        self.clear()

    def clear(self):
        self.namespaces = ["cv"]
        mat_class_info = ClassInfo([ 'class Mat', '', [], [] ], self.namespaces)
        mat_class_info.namespace = "cv"
        self.classes = { "Mat" : mat_class_info }
        self.classes["Mat"].namespace = "cv"
        self.module = ""
        self.Module = ""
        self.extension_implementations = None # Swift extensions implementations stream
        self.ported_func_list = []
        self.skipped_func_list = []
        self.def_args_hist = {} # { def_args_cnt : funcs_cnt }

    def add_class(self, decl):
        classinfo = ClassInfo(decl, namespaces=self.namespaces)
        if classinfo.name in class_ignore_list:
            logging.info('ignored: %s', classinfo)
            return None
        if classinfo.name != self.Module:
            self.classes[self.Module].member_classes.append(classinfo.objc_name)
        name = classinfo.cname
        if self.isWrapped(name) and not classinfo.base:
            logging.warning('duplicated: %s', classinfo)
            return None
        if name in self.classes:  # TODO implement inner namespaces
            if self.classes[name].symbol_id != classinfo.symbol_id:
                logging.warning('duplicated under new id: {} (was {})'.format(classinfo.symbol_id, self.classes[name].symbol_id))
                return None
        self.classes[name] = classinfo
        if name in type_dict and not classinfo.base:
            logging.warning('duplicated: %s', classinfo)
            return None
        if name != self.Module:
            type_dict.setdefault(name, {}).update(
                { "objc_type" : classinfo.objc_name + "*",
                  "from_cpp" : "[" + classinfo.objc_name + " fromNative:%(n)s]",
                  "to_cpp" : "*(%(n)s." + classinfo.native_ptr_name + ")" }
            )

        # missing_consts { Module : { public : [[name, val],...], private : [[]...] } }
        if name in missing_consts:
            if 'public' in missing_consts[name]:
                for (n, val) in missing_consts[name]['public']:
                    classinfo.consts.append( ConstInfo([n, val], addedManually=True) )

        # class props
        for p in decl[3]:
            classinfo.props.append( ClassPropInfo(p) )

        if name != self.Module:
            type_dict.setdefault("Ptr_"+name, {}).update(
                { "objc_type" : classinfo.objc_name + "*",
                  "c_type" : name,
                  "real_c_type" : classinfo.real_cname,
                  "to_cpp": "%(n)s." + classinfo.native_ptr_name,
                  "from_cpp": "[" + name + " fromNative:%(n)s]"}
            )

        logging.info('ok: class %s, name: %s, base: %s', classinfo, name, classinfo.base)
        return classinfo

    def add_const(self, decl, scope=None, enumType=None): # [ "const cname", val, [], [] ]
        constinfo = ConstInfo(decl, namespaces=self.namespaces, enumType=enumType)
        if constinfo.isIgnored():
            logging.info('ignored: %s', constinfo)
        else:
            objc_type = enumType.rsplit(".", 1)[-1] if enumType else ""
            if constinfo.enumType and constinfo.classpath:
                new_name = constinfo.classname + '_' + constinfo.name
                const_fix.setdefault(constinfo.classpath, {}).setdefault(objc_type, {})[constinfo.name] = new_name
                constinfo.swift_name = constinfo.name
                constinfo.name = new_name
                logging.info('use outer class prefix: %s', constinfo)

            if constinfo.classpath in const_fix and objc_type in const_fix[constinfo.classpath]:
                fixed_consts = const_fix[constinfo.classpath][objc_type]
                if constinfo.name in fixed_consts:
                    fixed_const = fixed_consts[constinfo.name]
                    constinfo.name = fixed_const
                    constinfo.cname = fixed_const
                if constinfo.value in fixed_consts:
                    constinfo.value = fixed_consts[constinfo.value]

            if not self.isWrapped(constinfo.classname):
                logging.info('class not found: %s', constinfo)
                if not constinfo.name.startswith(constinfo.classname + "_"):
                    constinfo.swift_name = constinfo.name
                    constinfo.name = constinfo.classname + '_' + constinfo.name
                constinfo.classname = ''

            ci = self.getClass(constinfo.classname)
            duplicate = ci.getConst(constinfo.name)
            if duplicate:
                if duplicate.addedManually:
                    logging.info('manual: %s', constinfo)
                else:
                    logging.warning('duplicated: %s', constinfo)
            else:
                ci.addConst(constinfo)
                logging.info('ok: %s', constinfo)

    def add_enum(self, decl): # [ "enum cname", "", [], [] ]
        enumType = decl[0].rsplit(" ", 1)[1]
        if enumType.endswith("<unnamed>"):
            enumType = None
        else:
            ctype = normalize_class_name(enumType)
            constinfo = ConstInfo(decl[3][0], namespaces=self.namespaces, enumType=enumType)
            objc_type = enumType.rsplit(".", 1)[-1]
            if objc_type in enum_ignore_list:
                return
            if constinfo.classname in enum_fix:
                objc_type = enum_fix[constinfo.classname].get(objc_type, objc_type)
            import_module = constinfo.classname if constinfo.classname and constinfo.classname != objc_type else self.Module
            type_dict[ctype] = { "cast_from" : "int",
                                 "cast_to" : get_cname(enumType),
                                 "objc_type" : objc_type,
                                 "is_enum" : True,
                                 "import_module" : import_module,
                                 "from_cpp" : "(" + objc_type + ")%(n)s"}
            type_dict[objc_type] = { "cast_to" : get_cname(enumType),
                                     "objc_type": objc_type,
                                     "is_enum": True,
                                     "import_module": import_module,
                                     "from_cpp": "(" + objc_type + ")%(n)s"}
            self.classes[self.Module].member_enums.append(objc_type)

        const_decls = decl[3]

        for decl in const_decls:
            self.add_const(decl, self.Module, enumType)

    def add_func(self, decl):
        fi = FuncInfo(decl, self.Module, namespaces=self.namespaces)
        classname = fi.classname or self.Module
        if classname in class_ignore_list:
            logging.info('ignored: %s', fi)
        elif classname in ManualFuncs and fi.objc_name in ManualFuncs[classname]:
            logging.info('manual: %s', fi)
            if "objc_method_name" in ManualFuncs[classname][fi.objc_name]:
                method_dict[(classname, fi.objc_name)] = ManualFuncs[classname][fi.objc_name]["objc_method_name"]
        elif not self.isWrapped(classname):
            logging.warning('not found: %s', fi)
        else:
            ci = self.getClass(classname)
            if ci.symbol_id != fi.symbol_id[0:fi.symbol_id.rfind('.')] and ci.symbol_id != self.Module:
                # TODO fix this (inner namepaces)
                logging.warning('SKIP: mismatched class: {} (class: {})'.format(fi.symbol_id, ci.symbol_id))
                return
            ci.addMethod(fi)
            logging.info('ok: %s', fi)
            # calc args with def val
            cnt = len([a for a in fi.args if a.defval])
            self.def_args_hist[cnt] = self.def_args_hist.get(cnt, 0) + 1
            add_method_to_dict(classname, fi)

    def save(self, path, buf):
        global total_files, updated_files
        if len(buf) == 0:
            return
        total_files += 1
        if os.path.exists(path):
            with open(path, "rt") as f:
                content = f.read()
                if content == buf:
                    return
        with codecs.open(path, "w", "utf-8") as f:
            f.write(buf)
        updated_files += 1

    def get_namespace_prefix(self, cname):
        namespace = self.classes[cname].namespace if cname in self.classes else "cv"
        return namespace.replace(".", "::") + "::"

    def gen(self, srcfiles, module, output_path, output_objc_path, common_headers, manual_classes):
        self.clear()
        self.module = module
        self.objcmodule = make_objcmodule(module)
        self.Module = module.capitalize()
        extension_implementations = StringIO() # Swift extensions implementations stream
        extension_signatures = []

        # TODO: support UMat versions of declarations (implement UMat-wrapper for Java)
        parser = hdr_parser.CppHeaderParser(generate_umat_decls=False)

        module_ci = self.add_class( ['class ' + self.Module, '', [], []]) # [ 'class/struct cname', ':bases', [modlist] [props] ]
        module_ci.header_import = module + '.hpp'

        # scan the headers and build more descriptive maps of classes, consts, functions
        includes = []
        for hdr in common_headers:
            logging.info("\n===== Common header : %s =====", hdr)
            includes.append(header_import(hdr))
        for hdr in srcfiles:
            decls = parser.parse(hdr)
            self.namespaces = sorted(parser.namespaces)
            logging.info("\n\n===== Header: %s =====", hdr)
            logging.info("Namespaces: %s", sorted(parser.namespaces))
            if decls:
                includes.append(header_import(hdr))
            else:
                logging.info("Ignore header: %s", hdr)
            for decl in decls:
                logging.info("\n--- Incoming ---\n%s", pformat(decl[:5], 4)) # without docstring
                name = decl[0]
                try:
                    if name.startswith("struct") or name.startswith("class"):
                        ci = self.add_class(decl)
                        if ci:
                            ci.header_import = header_import(hdr)
                    elif name.startswith("const"):
                        self.add_const(decl)
                    elif name.startswith("enum"):
                        # enum
                        self.add_enum(decl)
                    else: # function
                        self.add_func(decl)
                except SkipSymbolException as e:
                    logging.info('SKIP: {} due to {}'.format(name, e))
        self.classes[self.Module].member_classes += manual_classes

        logging.info("\n\n===== Generating... =====")
        package_path = os.path.join(output_objc_path, self.objcmodule)
        mkdir_p(package_path)
        extension_file = "%s/%sExt.swift" % (package_path, make_objcname(self.Module))

        for ci in sorted(self.classes.values(), key=lambda x: x.symbol_id):
            if ci.name == "Mat":
                continue
            ci.initCodeStreams(self.Module)
            self.gen_class(ci, self.module, extension_implementations, extension_signatures)
            classObjcHeaderCode = ci.generateObjcHeaderCode(self.module, self.Module, ci.objc_name)
            objc_mangled_name = make_objcname(ci.objc_name)
            header_file = "%s/%s.h" % (package_path, objc_mangled_name)
            self.save(header_file, classObjcHeaderCode)
            self.header_files.append(header_file)
            classObjcBodyCode = ci.generateObjcBodyCode(self.module, self.Module)
            self.save("%s/%s.mm" % (package_path, objc_mangled_name), classObjcBodyCode)
            ci.cleanupCodeStreams()
        self.save(extension_file, extension_implementations.getvalue())
        extension_implementations.close()
        self.save(os.path.join(output_path, self.objcmodule+".txt"), self.makeReport())

    def makeReport(self):
        '''
        Returns string with generator report
        '''
        report = StringIO()
        total_count = len(self.ported_func_list)+ len(self.skipped_func_list)
        report.write("PORTED FUNCs LIST (%i of %i):\n\n" % (len(self.ported_func_list), total_count))
        report.write("\n".join(self.ported_func_list))
        report.write("\n\nSKIPPED FUNCs LIST (%i of %i):\n\n" % (len(self.skipped_func_list), total_count))
        report.write("".join(self.skipped_func_list))
        for i in sorted(self.def_args_hist.keys()):
            report.write("\n%i def args - %i funcs" % (i, self.def_args_hist[i]))
        return report.getvalue()

    def fullTypeName(self, t):
        if not type_dict[t].get("is_primitive", False) or "cast_to" in type_dict[t]:
            if "cast_to" in type_dict[t]:
                return type_dict[t]["cast_to"]
            else:
                namespace_prefix = self.get_namespace_prefix(t)
                return namespace_prefix + t
        else:
            return t

    def build_objc2cv_prologue(self, prologue, vector_type, vector_full_type, objc_type, vector_name, array_name):
        if not (vector_type in type_dict and "to_cpp" in type_dict[vector_type] and type_dict[vector_type]["to_cpp"] != "%(n)s.nativeRef"):
            prologue.append("OBJC2CV(" + vector_full_type + ", " + objc_type[:-1] + ", " + vector_name + ", " + array_name + ");")
        else:
            conv_macro = "CONV_" + array_name
            prologue.append("#define " + conv_macro + "(e) " + type_dict[vector_type]["to_cpp"] % {"n": "e"})
            prologue.append("OBJC2CV_CUSTOM(" + vector_full_type + ", " + objc_type[:-1] + ", " + vector_name + ", " + array_name + ", " + conv_macro + ");")
            prologue.append("#undef " + conv_macro)

    def build_cv2objc_epilogue(self, epilogue, vector_type, vector_full_type, objc_type, vector_name, array_name):
        if not (vector_type in type_dict and "from_cpp" in type_dict[vector_type] and type_dict[vector_type]["from_cpp"] != ("[" + objc_type[:-1] + " fromNative:%(n)s]")):
            epilogue.append("CV2OBJC(" + vector_full_type + ", " + objc_type[:-1] + ", " + vector_name + ", " + array_name + ");")
        else:
            unconv_macro = "UNCONV_" + array_name
            epilogue.append("#define " + unconv_macro + "(e) " + type_dict[vector_type]["from_cpp"] % {"n": "e"})
            epilogue.append("CV2OBJC_CUSTOM(" + vector_full_type + ", " + objc_type[:-1] + ", " + vector_name + ", " + array_name + ", " + unconv_macro + ");")
            epilogue.append("#undef " + unconv_macro)

    def gen_func(self, ci, fi, extension_implementations, extension_signatures):
        logging.info("%s", fi)
        method_declarations = ci.method_declarations
        method_implementations = ci.method_implementations

        decl_args = []
        for a in fi.args:
            s = a.ctype or ' _hidden_ '
            if a.pointer:
                s += "*"
            elif a.out:
                s += "&"
            s += " " + a.name
            if a.defval:
                s += " = " + str(a.defval)
            decl_args.append(s)
        c_decl = "%s %s %s(%s)" % ( fi.static, fi.ctype, fi.cname, ", ".join(decl_args) )

        # comment
        method_declarations.write( "\n//\n// %s\n//\n" % c_decl )
        method_implementations.write( "\n//\n// %s\n//\n" % c_decl )
        # check if we 'know' all the types
        if fi.ctype not in type_dict: # unsupported ret type
            msg = "// Return type '%s' is not supported, skipping the function\n\n" % fi.ctype
            self.skipped_func_list.append(c_decl + "\n" + msg)
            method_declarations.write( " "*4 + msg )
            logging.warning("SKIP:" + c_decl.strip() + "\t due to RET type " + fi.ctype)
            return
        for a in fi.args:
            if a.ctype not in type_dict:
                if not a.defval and a.ctype.endswith("*"):
                    a.defval = 0
                if a.defval:
                    a.ctype = ''
                    continue
                msg = "// Unknown type '%s' (%s), skipping the function\n\n" % (a.ctype, a.out or "I")
                self.skipped_func_list.append(c_decl + "\n" + msg)
                method_declarations.write( msg )
                logging.warning("SKIP:" + c_decl.strip() + "\t due to ARG type " + a.ctype + "/" + (a.out or "I"))
                return

        self.ported_func_list.append(c_decl)

        # args
        args = fi.args[:] # copy
        objc_signatures=[]
        while True:
            # method args
            cv_args = []
            prologue = []
            epilogue = []
            if fi.ctype:
                ci.addImports(fi.ctype, False)
            for a in args:
                if not "v_type" in type_dict[a.ctype] and not "v_v_type" in type_dict[a.ctype]:
                    cast = ("(" + type_dict[a.ctype]["cast_to"] + ")") if "cast_to" in type_dict[a.ctype] else ""
                    cv_name = type_dict[a.ctype].get("to_cpp", cast + "%(n)s") if a.ctype else a.defval
                    if a.pointer and not cv_name == "0":
                        cv_name = "&(" + cv_name + ")"
                    if "O" in a.out and type_dict[a.ctype].get("out_type", ""):
                        cv_name = type_dict[a.ctype].get("out_type_ptr" if a.pointer else "out_type_ref", "%(n)s")
                    cv_args.append(type_dict[a.ctype].get("cv_name", cv_name) % {"n": a.name})
                    if not a.ctype: # hidden
                        continue
                    ci.addImports(a.ctype, "O" in a.out)
                if "v_type" in type_dict[a.ctype]: # pass as vector
                    vector_cpp_type = type_dict[a.ctype]["v_type"]
                    objc_type = type_dict[a.ctype]["objc_type"]
                    has_namespace = vector_cpp_type.find("::") != -1
                    ci.addImports(a.ctype, False)
                    vector_full_cpp_type = self.fullTypeName(vector_cpp_type) if not has_namespace else vector_cpp_type
                    vector_cpp_name = a.name + "Vector"
                    cv_args.append(vector_cpp_name)
                    self.build_objc2cv_prologue(prologue, vector_cpp_type, vector_full_cpp_type, objc_type, vector_cpp_name, a.name)
                    if "O" in a.out:
                        self.build_cv2objc_epilogue(epilogue, vector_cpp_type, vector_full_cpp_type, objc_type, vector_cpp_name, a.name)

                if "v_v_type" in type_dict[a.ctype]: # pass as vector of vector
                    vector_cpp_type = type_dict[a.ctype]["v_v_type"]
                    objc_type = type_dict[a.ctype]["objc_type"]
                    ci.addImports(a.ctype, False)
                    vector_full_cpp_type = self.fullTypeName(vector_cpp_type)
                    vector_cpp_name = a.name + "Vector2"
                    cv_args.append(vector_cpp_name)
                    prologue.append("OBJC2CV2(" + vector_full_cpp_type + ", " + objc_type[:-1] + ", " + vector_cpp_name + ", " + a.name +  ");")
                    if "O" in a.out:
                        epilogue.append(
                            "CV2OBJC2(" + vector_full_cpp_type + ", " + objc_type[:-1] + ", " + vector_cpp_name + ", " + a.name + ");")

            # calculate method signature to check for uniqueness
            objc_args = build_objc_args(args)
            objc_signature = fi.signature(args)
            swift_ext = make_swift_extension(args)
            logging.info("Objective-C: " + objc_signature)

            if objc_signature in objc_signatures:
                if args:
                    args.pop()
                    continue
                else:
                    break

            # doc comment
            if fi.docstring:
                lines = fi.docstring.splitlines()
                toWrite = []
                for index, line in enumerate(lines):
                    p0 = line.find("@param")
                    if p0 != -1:
                        p0 += 7 # len("@param" + 1)
                        p1 = line.find(' ', p0)
                        p1 = len(line) if p1 == -1 else p1
                        name = line[p0:p1]
                        for arg in args:
                            if arg.name == name:
                                toWrite.append(re.sub(r'\*\s*@param ', '* @param ', line))
                                break
                    else:
                        s0 = line.find("@see")
                        if s0 != -1:
                            sees = line[(s0 + 5):].split(",")
                            toWrite.append(line[:(s0 + 5)] + ", ".join(["`" + see_lookup(ci.objc_name, see.strip()) + "`" for see in sees]))
                        else:
                            toWrite.append(line)

                for line in toWrite:
                    method_declarations.write(line + "\n")

            # public wrapper method impl (calling native one above)
            # e.g.
            # public static void add( Mat src1, Mat src2, Mat dst, Mat mask, int dtype )
            # { add_0( src1.nativeObj, src2.nativeObj, dst.nativeObj, mask.nativeObj, dtype );  }
            ret_type = fi.ctype
            if fi.ctype.endswith('*'):
                ret_type = ret_type[:-1]
            ret_val = self.fullTypeName(fi.ctype) + " retVal = "
            ret = "return retVal;"
            tail = ""
            constructor = False
            if "v_type" in type_dict[ret_type]:
                objc_type = type_dict[ret_type]["objc_type"]
                vector_type = type_dict[ret_type]["v_type"]
                full_cpp_type = (self.get_namespace_prefix(vector_type) if (vector_type.find("::") == -1) else "") + vector_type
                prologue.append("NSMutableArray<" + objc_type + ">* retVal = [NSMutableArray new];")
                ret_val = "std::vector<" + full_cpp_type + "> retValVector = "
                self.build_cv2objc_epilogue(epilogue, vector_type, full_cpp_type, objc_type, "retValVector", "retVal")
            elif "v_v_type" in type_dict[ret_type]:
                objc_type = type_dict[ret_type]["objc_type"]
                cpp_type = type_dict[ret_type]["v_v_type"]
                if cpp_type.find("::") == -1:
                    cpp_type = self.get_namespace_prefix(cpp_type) + cpp_type
                prologue.append("NSMutableArray<NSMutableArray<" + objc_type + ">*>* retVal = [NSMutableArray new];")
                ret_val = "std::vector< std::vector<" + cpp_type + "> > retValVector = "
                epilogue.append("CV2OBJC2(" + cpp_type + ", " + objc_type[:-1] + ", retValVector, retVal);")
            elif ret_type.startswith("Ptr_"):
                cpp_type = type_dict[ret_type]["c_type"]
                real_cpp_type = type_dict[ret_type].get("real_c_type", cpp_type)
                namespace_prefix = self.get_namespace_prefix(cpp_type)
                ret_val = "cv::Ptr<" + namespace_prefix + real_cpp_type + "> retVal = "
                ret = "return [" + type_dict[ret_type]["objc_type"][:-1] + " fromNative:retVal];"
            elif ret_type == "void":
                ret_val = ""
                ret = ""
            elif ret_type == "": # c-tor
                constructor = True
                ret_val = "return [self initWithNativePtr:cv::Ptr<" + fi.fullClass(isCPP=True) + ">(new "
                tail = ")]"
                ret = ""
            elif self.isWrapped(ret_type): # wrapped class
                namespace_prefix = self.get_namespace_prefix(ret_type)
                ret_val = "cv::Ptr<" + namespace_prefix + ret_type + "> retVal = new " + namespace_prefix + ret_type + "("
                tail = ")"
                ret_type_dict = type_dict[ret_type]
                from_cpp = ret_type_dict["from_cpp_ptr"] if "from_cpp_ptr" in ret_type_dict else ret_type_dict["from_cpp"]
                ret = "return " + (from_cpp % { "n" : "retVal" }) + ";"
            elif "from_cpp" in type_dict[ret_type]:
                ret = "return " + (type_dict[ret_type]["from_cpp"] % { "n" : "retVal" }) + ";"

            static = fi.static if fi.classname else True

            objc_ret_type = type_dict[fi.ctype]["objc_type"] if type_dict[fi.ctype]["objc_type"] else "void" if not constructor else "instancetype"
            if "v_type" in type_dict[ret_type]:
                objc_ret_type = "NSArray<" + objc_ret_type + ">*"
            elif "v_v_type" in type_dict[ret_type]:
                objc_ret_type = "NSArray<NSArray<" + objc_ret_type + ">*>*"

            prototype = Template("$static ($objc_ret_type)$objc_name$objc_args").substitute(
                    static = "+" if static else "-",
                    objc_ret_type = objc_ret_type,
                    objc_args = " ".join(objc_args),
                    objc_name = fi.objc_name if not constructor else ("init" + ("With" + (args[0].name[0].upper() + args[0].name[1:]) if len(args) > 0 else ""))
                )

            if fi.prolog is not None:
                method_declarations.write("\n%s\n\n" % fi.prolog)

            method_declarations.write( Template(
"""$prototype$swift_name$deprecation_decl;

"""
                ).substitute(
                    prototype = prototype,
                    swift_name = " NS_SWIFT_NAME(" + fi.swift_name + "(" + build_swift_signature(args) + "))" if not constructor else "",
                    deprecation_decl = " DEPRECATED_ATTRIBUTE" if fi.deprecated else ""
                )
            )

            if fi.epilog is not None:
                method_declarations.write("%s\n\n" % fi.epilog)

            method_implementations.write( Template(
"""$prototype {$prologue
    $ret_val$obj_deref$cv_name($cv_args)$tail;$epilogue$ret
}

"""
                ).substitute(
                    prototype = prototype,
                    ret = "\n    " + ret if ret else "",
                    ret_val = ret_val,
                    prologue = "\n    " + "\n    ".join(prologue) if prologue else "",
                    epilogue = "\n    " + "\n    ".join(epilogue) if epilogue else "",
                    static = "+" if static else "-",
                    obj_deref = ("self." + ci.native_ptr_name + "->") if not static and not constructor else "",
                    cv_name = fi.cv_name if static else fi.fullClass(isCPP=True) if constructor else fi.name,
                    cv_args = ", ".join(cv_args),
                    tail = tail
                )
            )

            if swift_ext:
                prototype = build_swift_extension_decl(fi.swift_name, args, constructor, static, ret_type)
                if not (ci.name, prototype) in extension_signatures and not (ci.base, prototype) in extension_signatures:
                    (pro, epi) = build_swift_logues(args)
                    extension_implementations.write( Template(
"""public extension $classname {
    $deprecation_decl$prototype {
$prologue
$unrefined_call$epilogue$ret
    }
}

"""
                        ).substitute(
                            classname = make_objcname(ci.name),
                            deprecation_decl = "@available(*, deprecated)\n    " if fi.deprecated else "",
                            prototype = prototype,
                            prologue = "        " + "\n        ".join(pro),
                            unrefined_call = "        " + build_unrefined_call(fi.swift_name, args, constructor, static, ci.name, ret_type is not None and ret_type != "void"),
                            epilogue = "\n        " + "\n        ".join(epi) if len(epi) > 0 else "",
                            ret = "\n        return ret" if ret_type is not None and ret_type != "void" and not constructor else ""
                        )
                    )
                extension_signatures.append((ci.name, prototype))

            # adding method signature to dictionary
            objc_signatures.append(objc_signature)

            # processing args with default values
            if args and args[-1].defval:
                args.pop()
            else:
                break

    def gen_class(self, ci, module, extension_implementations, extension_signatures):
        logging.info("%s", ci)
        additional_imports = []
        if module in AdditionalImports:
            if "*" in AdditionalImports[module]:
                additional_imports += AdditionalImports[module]["*"]
            if ci.name in AdditionalImports[module]:
                additional_imports += AdditionalImports[module][ci.name]
        if hasattr(ci, 'header_import'):
            h = '"{}"'.format(ci.header_import)
            if not h in additional_imports:
                additional_imports.append(h)

        h = '"{}.hpp"'.format(module)
        if h in additional_imports:
            additional_imports.remove(h)
        h = '"opencv2/{}.hpp"'.format(module)
        if not h in additional_imports:
            additional_imports.insert(0, h)

        if additional_imports:
            ci.additionalImports.write('\n'.join(['#import %s' % make_objcname(h) for h in additional_imports]))

        # constants
        wrote_consts_pragma = False
        consts_map = {c.name: c for c in ci.private_consts}
        consts_map.update({c.name: c for c in ci.consts})
        def const_value(v):
            if v in consts_map:
                target = consts_map[v]
                assert target.value != v
                return const_value(target.value)
            return v
        if ci.consts:
            enumTypes = set([c.enumType for c in ci.consts])
            grouped_consts = {enumType: [c for c in ci.consts if c.enumType == enumType] for enumType in enumTypes}
            for typeName in sorted(grouped_consts.keys(), key=lambda x: str(x) if x is not None else ""):
                consts = grouped_consts[typeName]
                logging.info("%s", consts)
                if typeName:
                    typeNameShort = typeName.rsplit(".", 1)[-1]
                    if ci.cname in enum_fix:
                        typeNameShort = enum_fix[ci.cname].get(typeNameShort, typeNameShort)

                    ci.enum_declarations.write("""
// C++: enum {1} ({2})
typedef NS_ENUM(int, {1}) {{
    {0}\n}};\n\n""".format(
                        ",\n    ".join(["%s = %s" % (c.name + (" NS_SWIFT_NAME(" + c.swift_name + ")" if c.swift_name else ""), c.value) for c in consts]),
                        typeNameShort, typeName)
                    )
                else:
                    if not wrote_consts_pragma:
                        ci.method_declarations.write("#pragma mark - Class Constants\n\n")
                        wrote_consts_pragma = True
                    ci.method_declarations.write("""
{0}\n\n""".format("\n".join(["@property (class, readonly) int %s NS_SWIFT_NAME(%s);" % (c.name, c.name) for c in consts]))
                    )
                    declared_consts = []
                    match_alphabet = re.compile("[a-zA-Z]")
                    for c in consts:
                        value = str(c.value)
                        if match_alphabet.search(value):
                            for declared_const in sorted(declared_consts, key=len, reverse=True):
                                regex = re.compile("(?<!" + ci.cname + ".)" + declared_const)
                                value = regex.sub(ci.cname + "." + declared_const, value)
                        ci.method_implementations.write("+ (int)%s {\n    return %s;\n}\n\n" % (c.name, value))
                        declared_consts.append(c.name)

        ci.method_declarations.write("#pragma mark - Methods\n\n")

        # methods
        for fi in ci.getAllMethods():
            self.gen_func(ci, fi, extension_implementations, extension_signatures)
        # props
        for pi in ci.props:
            ci.method_declarations.write("\n    //\n    // C++: %s %s::%s\n    //\n\n" % (pi.ctype, ci.fullName(isCPP=True), pi.name))
            type_data = type_dict[pi.ctype] if pi.ctype != "uchar" else {"objc_type" : "unsigned char", "is_primitive" : True}
            objc_type = type_data.get("objc_type", pi.ctype)
            ci.addImports(pi.ctype, False)
            ci.method_declarations.write("@property " + ("(readonly) " if not pi.rw else "") + objc_type + " " + pi.name + ";\n")
            ptr_ref = "self." + ci.native_ptr_name + "->" if not ci.is_base_class else "self.nativePtr->"
            if "v_type" in type_data:
                vector_cpp_type = type_data["v_type"]
                has_namespace = vector_cpp_type.find("::") != -1
                vector_full_cpp_type = self.fullTypeName(vector_cpp_type) if not has_namespace else vector_cpp_type
                ret_val = "std::vector<" + vector_full_cpp_type + "> retValVector = "
                ci.method_implementations.write("-(NSArray<" + objc_type + ">*)" + pi.name + " {\n")
                ci.method_implementations.write("\tNSMutableArray<" + objc_type + ">* retVal = [NSMutableArray new];\n")
                ci.method_implementations.write("\t" + ret_val + ptr_ref + pi.name + ";\n")
                epilogue = []
                self.build_cv2objc_epilogue(epilogue, vector_cpp_type, vector_full_cpp_type, objc_type, "retValVector", "retVal")
                ci.method_implementations.write("\t" + ("\n\t".join(epilogue)) + "\n")
                ci.method_implementations.write("\treturn retVal;\n}\n\n")
            elif "v_v_type" in type_data:
                vector_cpp_type = type_data["v_v_type"]
                has_namespace = vector_cpp_type.find("::") != -1
                vector_full_cpp_type = self.fullTypeName(vector_cpp_type) if not has_namespace else vector_cpp_type
                ret_val = "std::vector<std::vector<" + vector_full_cpp_type + ">> retValVectorVector = "
                ci.method_implementations.write("-(NSArray<NSArray<" + objc_type + ">*>*)" + pi.name + " {\n")
                ci.method_implementations.write("\tNSMutableArray<NSMutableArray<" + objc_type + ">*>* retVal = [NSMutableArray new];\n")
                ci.method_implementations.write("\t" + ret_val + ptr_ref + pi.name + ";\n")
                ci.method_implementations.write("\tCV2OBJC2(" + vector_full_cpp_type + ", " + objc_type[:-1] + ", retValVectorVector, retVal);\n")
                ci.method_implementations.write("\treturn retVal;\n}\n\n")
            elif self.isWrapped(pi.ctype):  # wrapped class
                namespace_prefix = self.get_namespace_prefix(pi.ctype)
                ci.method_implementations.write("-(" + objc_type + ")" + pi.name + " {\n")
                ci.method_implementations.write("\tcv::Ptr<" + namespace_prefix + pi.ctype + "> retVal = new " + namespace_prefix + pi.ctype + "(" + ptr_ref + pi.name + ");\n")
                from_cpp = type_data["from_cpp_ptr"] if "from_cpp_ptr" in type_data else type_data["from_cpp"]
                ci.method_implementations.write("\treturn " + (from_cpp % {"n": "retVal"}) + ";\n}\n\n")
            else:
                from_cpp = type_data.get("from_cpp", "%(n)s")
                retVal = from_cpp % {"n": (ptr_ref + pi.name)}
                ci.method_implementations.write("-(" + objc_type + ")" + pi.name + " {\n\treturn " + retVal + ";\n}\n\n")
            if pi.rw:
                if "v_type" in type_data:
                    vector_cpp_type = type_data["v_type"]
                    has_namespace = vector_cpp_type.find("::") != -1
                    vector_full_cpp_type = self.fullTypeName(vector_cpp_type) if not has_namespace else vector_cpp_type
                    ci.method_implementations.write("-(void)set" + pi.name[0].upper() + pi.name[1:] + ":(NSArray<" + objc_type + ">*)" + pi.name + "{\n")
                    prologue = []
                    self.build_objc2cv_prologue(prologue, vector_cpp_type, vector_full_cpp_type, objc_type, "valVector", pi.name)
                    ci.method_implementations.write("\t" + ("\n\t".join(prologue)) + "\n")
                    ci.method_implementations.write("\t" + ptr_ref + pi.name + " = valVector;\n}\n\n")
                else:
                    to_cpp = type_data.get("to_cpp", ("(" + type_data.get("cast_to") + ")%(n)s") if "cast_to" in type_data else "%(n)s")
                    val = to_cpp % {"n": pi.name}
                    ci.method_implementations.write("-(void)set" + pi.name[0].upper() + pi.name[1:] + ":(" + objc_type + ")" + pi.name + " {\n\t" + ptr_ref + pi.name + " = " + val + ";\n}\n\n")

        # manual ports
        if ci.name in ManualFuncs:
            for func in sorted(ManualFuncs[ci.name].keys()):
                logging.info("manual function: %s", func)
                fn = ManualFuncs[ci.name][func]
                ci.method_declarations.write( "\n".join(fn["declaration"]) )
                ci.method_implementations.write( "\n".join(fn["implementation"]) )

    def getClass(self, classname):
        return self.classes[classname or self.Module]

    def isWrapped(self, classname):
        name = classname or self.Module
        return name in self.classes

    def isSmartClass(self, ci):
        '''
        Check if class stores Ptr<T>* instead of T* in nativeObj field
        '''
        if ci.smart != None:
            return ci.smart

        # if parents are smart (we hope) then children are!
        # if not we believe the class is smart if it has "create" method
        ci.smart = False
        if ci.base or ci.name == 'Algorithm':
            ci.smart = True
        else:
            for fi in ci.methods:
                if fi.name == "create":
                    ci.smart = True
                    break

        return ci.smart

    def smartWrap(self, ci, fullname):
        '''
        Wraps fullname with Ptr<> if needed
        '''
        if self.isSmartClass(ci):
            return "Ptr<" + fullname + ">"
        return fullname

    def finalize(self, objc_target, output_objc_path, output_objc_build_path):
        opencv_header_file = os.path.join(output_objc_path, framework_name + ".h")
        opencv_header = "#import <Foundation/Foundation.h>\n\n"
        opencv_header += "// ! Project version number\nFOUNDATION_EXPORT double " + framework_name + "VersionNumber;\n\n"
        opencv_header += "// ! Project version string\nFOUNDATION_EXPORT const unsigned char " + framework_name + "VersionString[];\n\n"
        opencv_header += "\n".join(["#define AVAILABLE_" + m['name'].upper() for m in config['modules']])
        opencv_header += "\n\n"
        opencv_header += "\n".join(["#import <" + framework_name + "/%s>" % os.path.basename(f) for f in self.header_files])
        self.save(opencv_header_file, opencv_header)
        opencv_modulemap_file = os.path.join(output_objc_path, framework_name + ".modulemap")
        opencv_modulemap = "framework module " + framework_name + " {\n"
        opencv_modulemap += "  umbrella header \"" + framework_name + ".h\"\n"
        opencv_modulemap += "\n".join(["  header \"%s\"" % os.path.basename(f) for f in self.header_files])
        opencv_modulemap += "\n  export *\n  module * {export *}\n}\n"
        self.save(opencv_modulemap_file, opencv_modulemap)
        available_modules = " ".join(["-DAVAILABLE_" + m['name'].upper() for m in config['modules']])
        cmakelist_template = read_contents(os.path.join(SCRIPT_DIR, 'templates/cmakelists.template'))
        cmakelist = Template(cmakelist_template).substitute(modules = ";".join(modules), framework = framework_name, objc_target=objc_target, module_availability_defines=available_modules)
        self.save(os.path.join(dstdir, "CMakeLists.txt"), cmakelist)
        mkdir_p(os.path.join(output_objc_build_path, "framework_build"))
        mkdir_p(os.path.join(output_objc_build_path, "test_build"))
        mkdir_p(os.path.join(output_objc_build_path, "doc_build"))
        with open(os.path.join(SCRIPT_DIR, '../doc/README.md')) as readme_in:
            readme_body = readme_in.read()
        readme_body += "\n\n\n##Modules\n\n" + ", ".join(["`" + m.capitalize() + "`" for m in modules])
        with open(os.path.join(output_objc_build_path, "doc_build/README.md"), "w") as readme_out:
            readme_out.write(readme_body)
        if framework_name != "OpenCV":
            for dirname, dirs, files in os.walk(os.path.join(testdir, "test")):
                if dirname.endswith('/resources'):
                    continue  # don't touch resource binary files
                for filename in files:
                    filepath = os.path.join(dirname, filename)
                    with io.open(filepath, encoding="utf-8", errors="ignore") as file:
                        body = file.read()
                    body = body.replace("import OpenCV", "import " + framework_name)
                    body = body.replace("#import <OpenCV/OpenCV.h>", "#import <" + framework_name + "/" + framework_name + ".h>")
                    with codecs.open(filepath, "w", "utf-8") as file:
                        file.write(body)


def copy_objc_files(objc_files_dir, objc_base_path, module_path, include = False):
    global total_files, updated_files
    objc_files = []
    re_filter = re.compile(r'^.+\.(h|m|mm|swift)$')
    for root, dirnames, filenames in os.walk(objc_files_dir):
       objc_files += [os.path.join(root, filename) for filename in filenames if re_filter.match(filename)]
    objc_files = [f.replace('\\', '/') for f in objc_files]

    re_prefix = re.compile(r'^.+/(.+)\.(h|m|mm|swift)$')
    for objc_file in objc_files:
        src = objc_file
        m = re_prefix.match(objc_file)
        target_fname = (m.group(1) + '.' + m.group(2)) if m else os.path.basename(objc_file)
        dest = os.path.join(objc_base_path, os.path.join(module_path, target_fname))
        mkdir_p(os.path.dirname(dest))
        total_files += 1
        if include and m.group(2) == 'h':
            generator.header_files.append(dest)
        if (not os.path.exists(dest)) or (os.stat(src).st_mtime - os.stat(dest).st_mtime > 1):
            copyfile(src, dest)
            updated_files += 1
    return objc_files

def unescape(str):
    return str.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")

def escape_underscore(str):
    return str.replace('_', '\\_')

def escape_texttt(str):
    return re.sub(re.compile('texttt{(.*?)}', re.DOTALL), lambda x: 'texttt{' + escape_underscore(x.group(1)) + '}', str)

def get_macros(tex):
    out = ""
    if re.search(r"\\fork\s*{", tex):
        out += "\\newcommand{\\fork}[4]{ \\left\\{ \\begin{array}{l l} #1 & \\text{#2}\\\\\\\\ #3 & \\text{#4}\\\\\\\\ \\end{array} \\right.} "
    if re.search(r"\\vecthreethree\s*{", tex):
        out += "\\newcommand{\\vecthreethree}[9]{ \\begin{bmatrix} #1 & #2 & #3\\\\\\\\ #4 & #5 & #6\\\\\\\\ #7 & #8 & #9 \\end{bmatrix} } "
    return out

def fix_tex(tex):
    macros = get_macros(tex)
    fix_escaping = escape_texttt(unescape(tex))
    return macros + fix_escaping

def sanitize_documentation_string(doc, type):
    if type == "class":
        doc = doc.replace("@param ", "")

    doc = re.sub(re.compile('`\\$\\$(.*?)\\$\\$`', re.DOTALL), lambda x: '`$$' + fix_tex(x.group(1)) + '$$`', doc)
    doc = re.sub(re.compile('\\\\f\\{align\\*\\}\\{?(.*?)\\\\f\\}', re.DOTALL), lambda x: '`$$\\begin{aligned} ' + fix_tex(x.group(1)) + ' \\end{aligned}$$`', doc)
    doc = re.sub(re.compile('\\\\f\\{equation\\*\\}\\{(.*?)\\\\f\\}', re.DOTALL), lambda x: '`$$\\begin{aligned} ' + fix_tex(x.group(1)) + ' \\end{aligned}$$`', doc)
    doc = re.sub(re.compile('\\\\f\\$(.*?)\\\\f\\$', re.DOTALL), lambda x: '`$$' + fix_tex(x.group(1)) + '$$`', doc)
    doc = re.sub(re.compile('\\\\f\\[(.*?)\\\\f\\]', re.DOTALL), lambda x: '`$$' + fix_tex(x.group(1)) + '$$`', doc)
    doc = re.sub(re.compile('\\\\f\\{(.*?)\\\\f\\}', re.DOTALL), lambda x: '`$$' + fix_tex(x.group(1)) + '$$`', doc)

    doc = doc.replace("@anchor", "") \
        .replace("@brief ", "").replace("\\brief ", "") \
        .replace("@cite", "CITE:") \
        .replace("@code{.cpp}", "<code>") \
        .replace("@code{.txt}", "<code>") \
        .replace("@code", "<code>") \
        .replace("@copydoc", "") \
        .replace("@copybrief", "") \
        .replace("@date", "") \
        .replace("@defgroup", "") \
        .replace("@details ", "") \
        .replace("@endcode", "</code>") \
        .replace("@endinternal", "") \
        .replace("@file", "") \
        .replace("@include", "INCLUDE:") \
        .replace("@ingroup", "") \
        .replace("@internal", "") \
        .replace("@overload", "") \
        .replace("@param[in]", "@param") \
        .replace("@param[out]", "@param") \
        .replace("@ref", "REF:") \
        .replace("@note", "NOTE:") \
        .replace("@returns", "@return") \
        .replace("@sa ", "@see ") \
        .replace("@snippet", "SNIPPET:") \
        .replace("@todo", "TODO:") \

    lines = doc.splitlines()

    in_code = False
    for i,line in enumerate(lines):
        if line.find("</code>") != -1:
            in_code = False
            lines[i] = line.replace("</code>", "")
        if in_code:
            lines[i] = unescape(line)
        if line.find("<code>") != -1:
            in_code = True
            lines[i] = line.replace("<code>", "")

    lines = list([x[x.find('*'):].strip() if x.lstrip().startswith("*") else x for x in lines])
    lines = list(["* " + x[1:].strip() if x.startswith("*") and x != "*" else x for x in lines])
    lines = list([x if x.startswith("*") else "* " + x if x and x != "*" else "*" for x in lines])

    hasValues = False
    for line in lines:
        if line != "*":
            hasValues = True
            break
    return "/**\n " + "\n ".join(lines) + "\n */" if hasValues else ""

if __name__ == "__main__":
    # initialize logger
    logging.basicConfig(filename='gen_objc.log', format=None, filemode='w', level=logging.INFO)
    handler = logging.StreamHandler()
    handler.setLevel(os.environ.get('LOG_LEVEL', logging.WARNING))
    logging.getLogger().addHandler(handler)

    # parse command line parameters
    import argparse
    arg_parser = argparse.ArgumentParser(description='OpenCV Objective-C Wrapper Generator')
    arg_parser.add_argument('-p', '--parser', required=True, help='OpenCV header parser')
    arg_parser.add_argument('-c', '--config', required=True, help='OpenCV modules config')
    arg_parser.add_argument('-t', '--target', required=True, help='Target (either ios or osx or visionos)')
    arg_parser.add_argument('-f', '--framework', required=True, help='Framework name')

    args=arg_parser.parse_args()

    # import header parser
    hdr_parser_path = os.path.abspath(args.parser)
    if hdr_parser_path.endswith(".py"):
        hdr_parser_path = os.path.dirname(hdr_parser_path)
    sys.path.append(hdr_parser_path)
    import hdr_parser

    with open(args.config) as f:
        config = json.load(f)

    ROOT_DIR = config['rootdir']; assert os.path.exists(ROOT_DIR)
    if 'objc_build_dir' in config:
        objc_build_dir = config['objc_build_dir']
        assert os.path.exists(objc_build_dir), objc_build_dir
    else:
        objc_build_dir = os.getcwd()

    dstdir = "./gen"
    testdir = "./test"
    objc_base_path = os.path.join(dstdir, 'objc'); mkdir_p(objc_base_path)
    objc_test_base_path = testdir; mkdir_p(objc_test_base_path)
    copy_objc_files(os.path.join(SCRIPT_DIR, '../test/test'), objc_test_base_path, 'test', False)
    copy_objc_files(os.path.join(SCRIPT_DIR, '../test/dummy'), objc_test_base_path, 'dummy', False)
    copyfile(os.path.join(SCRIPT_DIR, '../test/cmakelists.template'), os.path.join(objc_test_base_path, 'CMakeLists.txt'))

    # launch Objective-C Wrapper generator
    generator = ObjectiveCWrapperGenerator()

    gen_dict_files = []
    framework_name = args.framework

    print("Objective-C: Processing OpenCV modules: %d" % len(config['modules']))
    for e in config['modules']:
        (module, module_location) = (e['name'], os.path.join(ROOT_DIR, e['location']))
        logging.info("\n=== MODULE: %s (%s) ===\n" % (module, module_location))
        modules.append(module)

        module_imports = []
        srcfiles = []
        common_headers = []

        misc_location = os.path.join(module_location, 'misc/objc')

        srcfiles_fname = os.path.join(misc_location, 'filelist')
        if os.path.exists(srcfiles_fname):
            with open(srcfiles_fname) as f:
                srcfiles = [os.path.join(module_location, str(l).strip()) for l in f.readlines() if str(l).strip()]
        else:
            re_bad = re.compile(r'(private|.inl.hpp$|_inl.hpp$|.detail.hpp$|.details.hpp$|_winrt.hpp$|/cuda/|/legacy/)')
            # .h files before .hpp
            h_files = []
            hpp_files = []
            for root, dirnames, filenames in os.walk(os.path.join(module_location, 'include')):
               h_files += [os.path.join(root, filename) for filename in fnmatch.filter(filenames, '*.h')]
               hpp_files += [os.path.join(root, filename) for filename in fnmatch.filter(filenames, '*.hpp')]
            srcfiles = h_files + hpp_files
            # Use relative paths to avoid being affected by the name of the parent directory.
            # See https://github.com/opencv/opencv/issues/26712
            srcfiles = [f for f in srcfiles if not re_bad.search(os.path.relpath(f, module_location).replace('\\', '/'))]
        logging.info("\nFiles (%d):\n%s", len(srcfiles), pformat(srcfiles))

        common_headers_fname = os.path.join(misc_location, 'filelist_common')
        if os.path.exists(common_headers_fname):
            with open(common_headers_fname) as f:
                common_headers = [os.path.join(module_location, str(l).strip()) for l in f.readlines() if str(l).strip()]
        logging.info("\nCommon headers (%d):\n%s", len(common_headers), pformat(common_headers))

        gendict_fname = os.path.join(misc_location, 'gen_dict.json')
        module_source_map = {}
        if os.path.exists(gendict_fname):
            with open(gendict_fname) as f:
                gen_type_dict = json.load(f)
            namespace_ignore_list = gen_type_dict.get("namespace_ignore_list", [])
            class_ignore_list += gen_type_dict.get("class_ignore_list", [])
            enum_ignore_list += gen_type_dict.get("enum_ignore_list", [])
            const_ignore_list += gen_type_dict.get("const_ignore_list", [])
            const_private_list += gen_type_dict.get("const_private_list", [])
            missing_consts.update(gen_type_dict.get("missing_consts", {}))
            type_dict.update(gen_type_dict.get("type_dict", {}))
            AdditionalImports[module] = gen_type_dict.get("AdditionalImports", {})
            ManualFuncs.update(gen_type_dict.get("ManualFuncs", {}))
            func_arg_fix.update(gen_type_dict.get("func_arg_fix", {}))
            header_fix.update(gen_type_dict.get("header_fix", {}))
            enum_fix.update(gen_type_dict.get("enum_fix", {}))
            const_fix.update(gen_type_dict.get("const_fix", {}))
            module_source_map = gen_type_dict.get("SourceMap", {})
            namespaces_dict.update(gen_type_dict.get("namespaces_dict", {}))
            module_imports += gen_type_dict.get("module_imports", [])

        objc_files_dir = os.path.join(misc_location, 'common')
        copied_files = []
        if os.path.exists(objc_files_dir):
            copied_files += copy_objc_files(objc_files_dir, objc_base_path, module, True)

        target_path = 'macosx' if args.target == 'osx' else module_source_map.get(args.target, args.target)
        target_files_dir = os.path.join(misc_location, target_path)
        if os.path.exists(target_files_dir):
            copied_files += copy_objc_files(target_files_dir, objc_base_path, module, True)

        objc_test_files_dir = os.path.join(misc_location, 'test')
        if os.path.exists(objc_test_files_dir):
            copy_objc_files(objc_test_files_dir, objc_test_base_path, 'test', False)
            objc_test_resources_dir = os.path.join(objc_test_files_dir, 'resources')
            if os.path.exists(objc_test_resources_dir):
                copy_tree(objc_test_resources_dir, os.path.join(objc_test_base_path, 'test', 'resources'))

        manual_classes = [x for x in [x[x.rfind('/')+1:-2] for x in [x for x in copied_files if x.endswith('.h')]] if x in type_dict]

        if len(srcfiles) > 0:
            generator.gen(srcfiles, module, dstdir, objc_base_path, common_headers, manual_classes)
        else:
            logging.info("No generated code for module: %s", module)
    generator.finalize(args.target, objc_base_path, objc_build_dir)

    print('Generated files: %d (updated %d)' % (total_files, updated_files))
