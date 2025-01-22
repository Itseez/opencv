#!/usr/bin/env python

import sys, re, os.path, errno, fnmatch
import json
import logging
import codecs
from shutil import copyfile
from pprint import pformat
from string import Template

if sys.version_info[0] >= 3:
    from io import StringIO
else:
    import io
    class StringIO(io.StringIO):
        def write(self, s):
            if isinstance(s, str):
                s = unicode(s)  # noqa: F821
            return super(StringIO, self).write(s)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# list of modules + files remap
config = None
ROOT_DIR = None
FILES_REMAP = {}
def checkFileRemap(path):
    path = os.path.realpath(path)
    if path in FILES_REMAP:
        return FILES_REMAP[path]
    assert path[-3:] != '.in', path
    return path

total_files = 0
updated_files = 0

module_imports = []
module_j_code = None
module_jn_code = None

# list of class names, which should be skipped by wrapper generator
# the list is loaded from misc/java/gen_dict.json defined for the module and its dependencies
class_ignore_list = []

# list of constant names, which should be skipped by wrapper generator
# ignored constants can be defined using regular expressions
const_ignore_list = []

# list of private constants
const_private_list = []

# { Module : { public : [[name, val],...], private : [[]...] } }
missing_consts = {}

# c_type    : { java/jni correspondence }
# Complex data types are configured for each module using misc/java/gen_dict.json

type_dict = {
# "simple"  : { j_type : "?", jn_type : "?", jni_type : "?", suffix : "?" },
    ""        : { "j_type" : "", "jn_type" : "long", "jni_type" : "jlong" }, # c-tor ret_type
    "void"    : { "j_type" : "void", "jn_type" : "void", "jni_type" : "void" },
    "env"     : { "j_type" : "", "jn_type" : "", "jni_type" : "JNIEnv*"},
    "cls"     : { "j_type" : "", "jn_type" : "", "jni_type" : "jclass"},
    "bool"    : { "j_type" : "boolean", "jn_type" : "boolean", "jni_type" : "jboolean", "suffix" : "Z" },
    "char"    : { "j_type" : "char", "jn_type" : "char", "jni_type" : "jchar", "suffix" : "C" },
    "int"     : { "j_type" : "int", "jn_type" : "int", "jni_type" : "jint", "suffix" : "I" },
    "long"    : { "j_type" : "int", "jn_type" : "int", "jni_type" : "jint", "suffix" : "I" },
    "float"   : { "j_type" : "float", "jn_type" : "float", "jni_type" : "jfloat", "suffix" : "F" },
    "double"  : { "j_type" : "double", "jn_type" : "double", "jni_type" : "jdouble", "suffix" : "D" },
    "size_t"  : { "j_type" : "long", "jn_type" : "long", "jni_type" : "jlong", "suffix" : "J" },
    "__int64" : { "j_type" : "long", "jn_type" : "long", "jni_type" : "jlong", "suffix" : "J" },
    "int64"   : { "j_type" : "long", "jn_type" : "long", "jni_type" : "jlong", "suffix" : "J" },
    "double[]": { "j_type" : "double[]", "jn_type" : "double[]", "jni_type" : "jdoubleArray", "suffix" : "_3D" },
    'string'  : {  # std::string, see "String" in modules/core/misc/java/gen_dict.json
        'j_type': 'String',
        'jn_type': 'String',
        'jni_name': 'n_%(n)s',
        'jni_type': 'jstring',
        'jni_var': 'const char* utf_%(n)s = env->GetStringUTFChars(%(n)s, 0); std::string n_%(n)s( utf_%(n)s ? utf_%(n)s : "" ); env->ReleaseStringUTFChars(%(n)s, utf_%(n)s)',
        'suffix': 'Ljava_lang_String_2',
        'j_import': 'java.lang.String'
    },
    'vector_string': {  # std::vector<std::string>, see "vector_String" in modules/core/misc/java/gen_dict.json
        'j_type': 'List<String>',
        'jn_type': 'List<String>',
        'jni_type': 'jobject',
        'jni_var': 'std::vector< std::string > %(n)s',
        'suffix': 'Ljava_util_List',
        'v_type': 'string',
        'j_import': 'java.lang.String'
    },
}

# Defines a rule to add extra prefixes for names from specific namespaces.
# In example, cv::fisheye::stereoRectify from namespace fisheye is wrapped as fisheye_stereoRectify
namespaces_dict = {}

# { class : { func : {j_code, jn_code, cpp_code} } }
ManualFuncs = {}

# { class : { func : { arg_name : {"ctype" : ctype, "attrib" : [attrib]} } } }
func_arg_fix = {}

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

T_JAVA_START_INHERITED = read_contents(os.path.join(SCRIPT_DIR, 'templates/java_class_inherited.prolog'))
T_JAVA_START_ORPHAN = read_contents(os.path.join(SCRIPT_DIR, 'templates/java_class.prolog'))
T_JAVA_START_MODULE = read_contents(os.path.join(SCRIPT_DIR, 'templates/java_module.prolog'))
T_CPP_MODULE = Template(read_contents(os.path.join(SCRIPT_DIR, 'templates/cpp_module.template')))

class GeneralInfo():
    def __init__(self, type, decl, namespaces):
        self.symbol_id, self.parent_id, self.namespace, self.classpath, self.classname, self.name = self.parseName(decl[0], namespaces)
        self.cname = get_cname(self.symbol_id)

        # parse doxygen comments
        self.params={}
        self.annotation=[]
        if type == "class":
            docstring="// C++: class " + self.name + "\n"
        else:
            docstring=""

        if len(decl)>5 and decl[5]:
            doc = decl[5]

            #logging.info('docstring: %s', doc)
            if re.search("(@|\\\\)deprecated", doc):
                self.annotation.append("@Deprecated")

            docstring += sanitize_java_documentation_string(doc, type)

        self.docstring = docstring

    def parseName(self, name, namespaces):
        '''
        input: full name and available namespaces
        returns: (namespace, classpath, classname, name)
        '''
        name = name[name.find(" ")+1:].strip() # remove struct/class/const prefix
        parent = name[:name.rfind('.')].strip()
        if len(parent) == 0:
            parent = None
        spaceName = ""
        localName = name # <classes>.<name>
        for namespace in sorted(namespaces, key=len, reverse=True):
            if name.startswith(namespace + "."):
                spaceName = namespace
                localName = name.replace(namespace + ".", "")
                break
        pieces = localName.split(".")
        if len(pieces) > 2: # <class>.<class>.<class>.<name>
            return name, parent, spaceName, ".".join(pieces[:-1]), pieces[-2], pieces[-1]
        elif len(pieces) == 2: # <class>.<name>
            return name, parent, spaceName, pieces[0], pieces[0], pieces[1]
        elif len(pieces) == 1: # <name>
            return name, parent, spaceName, "", "", pieces[0]
        else:
            return name, parent, spaceName, "", "" # error?!

    def fullNameOrigin(self):
        result = self.symbol_id
        return result

    def fullNameJAVA(self):
        result = '.'.join([self.fullParentNameJAVA(), self.jname])
        return result

    def fullNameCPP(self):
        result = self.cname
        return result

    def fullParentNameJAVA(self):
        result = ".".join([f for f in [self.namespace] + self.classpath.split(".") if len(f)>0])
        return result

    def fullParentNameCPP(self):
        result = get_cname(self.parent_id)
        return result

class ConstInfo(GeneralInfo):
    def __init__(self, decl, addedManually=False, namespaces=[], enumType=None):
        GeneralInfo.__init__(self, "const", decl, namespaces)
        self.value = decl[1]
        self.enumType = enumType
        self.addedManually = addedManually
        if self.namespace in namespaces_dict:
            prefix = namespaces_dict[self.namespace]
            if prefix:
                self.name = '%s_%s' % (prefix, self.name)

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
        self.methods = []
        self.methods_suffixes = {}
        self.consts = [] # using a list to save the occurrence order
        self.private_consts = []
        self.imports = set()
        self.props= []
        self.jname = self.name
        self.smart = None # True if class stores Ptr<T>* instead of T* in nativeObj field
        self.j_code = None # java code stream
        self.jn_code = None # jni code stream
        self.cpp_code = None # cpp code stream
        for m in decl[2]:
            if m.startswith("="):
                self.jname = m[1:]
            if m == '/Simple':
                self.smart = False

        if self.classpath:
            prefix = self.classpath.replace('.', '_')
            self.name = '%s_%s' % (prefix, self.name)
            self.jname = '%s_%s' % (prefix, self.jname)

        if self.namespace in namespaces_dict:
            prefix = namespaces_dict[self.namespace]
            if prefix:
                self.name = '%s_%s' % (prefix, self.name)
                self.jname = '%s_%s' % (prefix, self.jname)

        self.base = ''
        if decl[1]:
            # FIXIT Use generator to find type properly instead of hacks below
            base_class = re.sub(r"^: ", "", decl[1])
            base_class = re.sub(r"^cv::", "", base_class)
            base_class = base_class.replace('::', '.')
            base_info = ClassInfo(('class {}'.format(base_class), '', [], [], None, None), [self.namespace])
            base_type_name = base_info.name
            if not base_type_name in type_dict:
                base_type_name = re.sub(r"^.*:", "", decl[1].split(",")[0]).strip().replace(self.jname, "")
            self.base = base_type_name
            self.addImports(self.base)

    def __repr__(self):
        return Template("CLASS $namespace::$classpath.$name : $base").substitute(**self.__dict__)

    def getAllImports(self, module):
        return ["import %s;" % c for c in sorted(self.imports) if not c.startswith('org.opencv.'+module)
            and (not c.startswith('java.lang.') or c.count('.') != 2)]

    def addImports(self, ctype):
        if ctype in type_dict:
            if "j_import" in type_dict[ctype]:
                self.imports.add(type_dict[ctype]["j_import"])
            if "v_type" in type_dict[ctype]:
                self.imports.add("java.util.List")
                self.imports.add("java.util.ArrayList")
                self.imports.add("org.opencv.utils.Converters")
                if type_dict[ctype]["v_type"] in ("Mat", "vector_Mat"):
                    self.imports.add("org.opencv.core.Mat")

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
        self.j_code = StringIO()
        self.jn_code = StringIO()
        self.cpp_code = StringIO()
        if self.base:
            self.j_code.write(T_JAVA_START_INHERITED)
        else:
            if self.name != Module:
                self.j_code.write(T_JAVA_START_ORPHAN)
            else:
                self.j_code.write(T_JAVA_START_MODULE)
        # misc handling
        if self.name == Module:
          for i in module_imports or []:
              self.imports.add(i)
          if module_j_code:
              self.j_code.write(module_j_code)
          if module_jn_code:
              self.jn_code.write(module_jn_code)

    def cleanupCodeStreams(self):
        self.j_code.close()
        self.jn_code.close()
        self.cpp_code.close()

    def generateJavaCode(self, m, M):
        return Template(self.j_code.getvalue() + "\n\n" +
                         self.jn_code.getvalue() + "\n}\n").substitute(
                            module = m,
                            name = self.name,
                            jname = self.jname,
                            jcleaner = "long nativeObjCopy = nativeObj;\n      org.opencv.core.Mat.cleaner.register(this, () -> delete(nativeObjCopy));",
                            imports = "\n".join(self.getAllImports(M)),
                            docs = self.docstring,
                            annotation = "\n" + "\n".join(self.annotation) if self.annotation else "",
                            base = self.base)

    def generateCppCode(self):
        return self.cpp_code.getvalue()

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
    def __init__(self, decl, namespaces=[]): # [ funcname, return_ctype, [modifiers], [args] ]
        GeneralInfo.__init__(self, "func", decl, namespaces)
        self.cname = get_cname(decl[0])
        self.jname = self.name
        self.isconstructor = self.name == self.classname
        if "[" in self.name:
            self.jname = "getelem"
        for m in decl[2]:
            if m.startswith("="):  # alias from WRAP_AS
                self.jname = m[1:]
        if self.classpath and self.classname != self.classpath:
            prefix = self.classpath.replace('.', '_')
            self.classname = prefix #'%s_%s' % (prefix, self.classname)
            if self.isconstructor:
                self.name = prefix #'%s_%s' % (prefix, self.name)
                self.jname = prefix #'%s_%s' % (prefix, self.jname)

        if self.namespace in namespaces_dict:
            prefix = namespaces_dict[self.namespace]
            if prefix:
                if self.classname:
                    self.classname = '%s_%s' % (prefix, self.classname)
                    if self.isconstructor:
                        self.jname = '%s_%s' % (prefix, self.jname)
                else:
                    self.jname = '%s_%s' % (prefix, self.jname)

        self.static = ["","static"][ "/S" in decl[2] ]
        self.ctype = re.sub(r"^CvTermCriteria", "TermCriteria", decl[1] or "")
        self.args = []
        func_fix_map = func_arg_fix.get(self.jname, {})
        for a in decl[3]:
            arg = a[:]
            arg_fix_map = func_fix_map.get(arg[1], {})
            arg[0] = arg_fix_map.get('ctype',  arg[0]) #fixing arg type
            arg[3] = arg_fix_map.get('attrib', arg[3]) #fixing arg attrib
            self.args.append(ArgInfo(arg))

    def fullClassJAVA(self):
        return self.fullParentNameJAVA()

    def fullClassCPP(self):
        return self.fullParentNameCPP()

    def __repr__(self):
        return Template("FUNC <$ctype $namespace.$classpath.$name $args>").substitute(**self.__dict__)

    def __lt__(self, other):
        return self.__repr__() < other.__repr__()


class JavaWrapperGenerator(object):
    def __init__(self):
        self.cpp_files = []
        self.clear()

    def clear(self):
        self.namespaces = ["cv"]
        classinfo_Mat = ClassInfo([ 'class cv.Mat', '', ['/Simple'], [] ], self.namespaces)
        self.classes = { "Mat" : classinfo_Mat }
        self.module = ""
        self.Module = ""
        self.ported_func_list = []
        self.skipped_func_list = []
        self.def_args_hist = {} # { def_args_cnt : funcs_cnt }

    def add_class(self, decl):
        classinfo = ClassInfo(decl, namespaces=self.namespaces)
        if classinfo.name in class_ignore_list:
            logging.info('ignored: %s', classinfo)
            return
        name = classinfo.name
        if self.isWrapped(name) and not classinfo.base:
            logging.warning('duplicated: %s', classinfo)
            return
        self.classes[name] = classinfo
        if name in type_dict and not classinfo.base:
            logging.warning('duplicated: %s', classinfo)
            return
        if self.isSmartClass(classinfo):
            jni_name = "*((*(Ptr<"+classinfo.fullNameCPP()+">*)%(n)s_nativeObj).get())"
        else:
            jni_name = "(*("+classinfo.fullNameCPP()+"*)%(n)s_nativeObj)"
        type_dict.setdefault(name, {}).update(
            { "j_type" : classinfo.jname,
              "jn_type" : "long", "jn_args" : (("__int64", ".nativeObj"),),
              "jni_name" : jni_name,
              "jni_type" : "jlong",
              "suffix" : "J",
              "j_import" : "org.opencv.%s.%s" % (self.module, classinfo.jname)
            }
        )
        type_dict.setdefault(name+'*', {}).update(
            { "j_type" : classinfo.jname,
              "jn_type" : "long", "jn_args" : (("__int64", ".nativeObj"),),
              "jni_name" : "&("+jni_name+")",
              "jni_type" : "jlong",
              "suffix" : "J",
              "j_import" : "org.opencv.%s.%s" % (self.module, classinfo.jname)
            }
        )

        # missing_consts { Module : { public : [[name, val],...], private : [[]...] } }
        if name in missing_consts:
            if 'private' in missing_consts[name]:
                for (n, val) in missing_consts[name]['private']:
                    classinfo.private_consts.append( ConstInfo([n, val], addedManually=True) )
            if 'public' in missing_consts[name]:
                for (n, val) in missing_consts[name]['public']:
                    classinfo.consts.append( ConstInfo([n, val], addedManually=True) )

        # class props
        for p in decl[3]:
            if True: #"vector" not in p[0]:
                classinfo.props.append( ClassPropInfo(p) )
            else:
                logging.warning("Skipped property: [%s]" % name, p)

        if classinfo.base:
            classinfo.addImports(classinfo.base)
        type_dict.setdefault("Ptr_"+name, {}).update(
            { "j_type" : classinfo.jname,
              "jn_type" : "long", "jn_args" : (("__int64", ".getNativeObjAddr()"),),
              "jni_name" : "*((Ptr<"+classinfo.fullNameCPP()+">*)%(n)s_nativeObj)", "jni_type" : "jlong",
              "suffix" : "J",
              "j_import" : "org.opencv.%s.%s" % (self.module, classinfo.jname)
            }
        )
        logging.info('ok: class %s, name: %s, base: %s', classinfo, name, classinfo.base)

    def add_const(self, decl, enumType=None): # [ "const cname", val, [], [] ]
        constinfo = ConstInfo(decl, namespaces=self.namespaces, enumType=enumType)
        if constinfo.isIgnored():
            logging.info('ignored: %s', constinfo)
        else:
            if not self.isWrapped(constinfo.classname):
                logging.info('class not found: %s', constinfo)
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
            type_dict[ctype] = { "cast_from" : "int", "cast_to" : get_cname(enumType), "j_type" : "int", "jn_type" : "int", "jni_type" : "jint", "suffix" : "I" }
        const_decls = decl[3]

        for decl in const_decls:
            self.add_const(decl, enumType)

    def add_func(self, decl):
        fi = FuncInfo(decl, namespaces=self.namespaces)
        classname = fi.classname or self.Module
        class_symbol_id = classname if self.isWrapped(classname) else fi.classpath.replace('.', '_') #('.'.join([fi.namespace, fi.classpath])[3:])
        if classname in class_ignore_list:
            logging.info('ignored: %s', fi)
        elif classname in ManualFuncs and fi.jname in ManualFuncs[classname]:
            logging.info('manual: %s', fi)
        elif not self.isWrapped(class_symbol_id):
            logging.warning('not found: %s', fi)
        else:
            self.getClass(class_symbol_id).addMethod(fi)
            logging.info('ok: %s', fi)
            # calc args with def val
            cnt = len([a for a in fi.args if a.defval])
            self.def_args_hist[cnt] = self.def_args_hist.get(cnt, 0) + 1

    def save(self, path, buf):
        global total_files, updated_files
        total_files += 1
        if os.path.exists(path):
            with open(path, "rt") as f:
                content = f.read()
                if content == buf:
                    return
        with codecs.open(path, "w", "utf-8") as f:
            f.write(buf)
        updated_files += 1

    def gen(self, srcfiles, module, output_path, output_jni_path, output_java_path, common_headers):
        self.clear()
        self.module = module
        self.Module = module.capitalize()
        # TODO: support UMat versions of declarations (implement UMat-wrapper for Java)
        parser = hdr_parser.CppHeaderParser(generate_umat_decls=False)

        self.add_class( ['class cv.' + self.Module, '', [], []] ) # [ 'class/struct cname', ':bases', [modlist] [props] ]

        # scan the headers and build more descriptive maps of classes, consts, functions
        includes = []
        for hdr in common_headers:
            logging.info("\n===== Common header : %s =====", hdr)
            includes.append('#include "' + hdr + '"')
        for hdr in srcfiles:
            decls = parser.parse(hdr)
            self.namespaces = sorted(parser.namespaces)
            logging.info("\n\n===== Header: %s =====", hdr)
            logging.info("Namespaces: %s", sorted(parser.namespaces))
            if decls:
                includes.append('#include "' + hdr + '"')
            else:
                logging.info("Ignore header: %s", hdr)
            for decl in decls:
                logging.info("\n--- Incoming ---\n%s", pformat(decl[:5], 4)) # without docstring
                name = decl[0]
                if name.startswith("struct") or name.startswith("class"):
                    self.add_class(decl)
                elif name.startswith("const"):
                    self.add_const(decl)
                elif name.startswith("enum"):
                    # enum
                    self.add_enum(decl)
                else: # function
                    self.add_func(decl)

        logging.info("\n\n===== Generating... =====")
        moduleCppCode = StringIO()
        package_path = os.path.join(output_java_path, module)
        mkdir_p(package_path)
        for ci in sorted(self.classes.values(), key=lambda x: x.symbol_id):
            if ci.name == "Mat":
                continue
            ci.initCodeStreams(self.Module)
            self.gen_class(ci)
            classJavaCode = ci.generateJavaCode(self.module, self.Module)
            self.save("%s/%s/%s.java" % (output_java_path, module, ci.jname), classJavaCode)
            moduleCppCode.write(ci.generateCppCode())
            ci.cleanupCodeStreams()
        cpp_file = os.path.abspath(os.path.join(output_jni_path, module + ".inl.hpp"))
        self.cpp_files.append(cpp_file)
        self.save(cpp_file, T_CPP_MODULE.substitute(m = module, M = module.upper(), code = moduleCppCode.getvalue(), includes = "\n".join(includes)))
        self.save(os.path.join(output_path, module+".txt"), self.makeReport())

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

    def fullTypeNameCPP(self, t):
        if self.isWrapped(t):
            return self.getClass(t).fullNameCPP()
        else:
            return cast_from(t)

    def gen_func(self, ci, fi, prop_name=''):
        logging.info("%s", fi)
        j_code   = ci.j_code
        jn_code  = ci.jn_code
        cpp_code = ci.cpp_code

        # c_decl
        # e.g: void add(Mat src1, Mat src2, Mat dst, Mat mask = Mat(), int dtype = -1)
        if prop_name:
            c_decl = "%s %s::%s" % (fi.ctype, fi.classname, prop_name)
        else:
            decl_args = []
            for a in fi.args:
                s = a.ctype or ' _hidden_ '
                if a.pointer:
                    s += "*"
                elif a.out:
                    s += "&"
                s += " " + a.name
                if a.defval:
                    s += " = "+a.defval
                decl_args.append(s)
            c_decl = "%s %s %s(%s)" % ( fi.static, fi.ctype, fi.cname, ", ".join(decl_args) )

        # java comment
        j_code.write( "\n    //\n    // C++: %s\n    //\n\n" % c_decl )
        # check if we 'know' all the types
        if fi.ctype not in type_dict: # unsupported ret type
            msg = "// Return type '%s' is not supported, skipping the function\n\n" % fi.ctype
            self.skipped_func_list.append(c_decl + "\n" + msg)
            j_code.write( " "*4 + msg )
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
                j_code.write( " "*4 + msg )
                logging.warning("SKIP:" + c_decl.strip() + "\t due to ARG type " + a.ctype + "/" + (a.out or "I"))
                return

        self.ported_func_list.append(c_decl)

        # jn & cpp comment
        jn_code.write( "\n    // C++: %s\n" % c_decl )
        cpp_code.write( "\n//\n// %s\n//\n" % c_decl )

        # java args
        args = fi.args[:] # copy
        j_signatures=[]
        suffix_counter = int(ci.methods_suffixes.get(fi.jname, -1))
        while True:
            suffix_counter += 1
            ci.methods_suffixes[fi.jname] = suffix_counter
             # java native method args
            jn_args = []
            # jni (cpp) function args
            jni_args = [ArgInfo([ "env", "env", "", [], "" ]), ArgInfo([ "cls", "", "", [], "" ])]
            j_prologue = []
            j_epilogue = []
            c_prologue = []
            c_epilogue = []
            if type_dict[fi.ctype]["jni_type"] == "jdoubleArray":
                fields = type_dict[fi.ctype]["jn_args"]
                c_epilogue.append( \
                    ("jdoubleArray _da_retval_ = env->NewDoubleArray(%(cnt)i);  " +
                     "jdouble _tmp_retval_[%(cnt)i] = {%(args)s}; " +
                     "env->SetDoubleArrayRegion(_da_retval_, 0, %(cnt)i, _tmp_retval_);") %
                    { "cnt" : len(fields), "args" : ", ".join(["(jdouble)_retval_" + f[1] for f in fields]) } )
            if fi.classname and fi.ctype and not fi.static: # non-static class method except c-tor
                # adding 'self'
                jn_args.append ( ArgInfo([ "__int64", "nativeObj", "", [], "" ]) )
                jni_args.append( ArgInfo([ "__int64", "self", "", [], "" ]) )
            ci.addImports(fi.ctype)
            for a in args:
                if not a.ctype: # hidden
                    continue
                ci.addImports(a.ctype)
                if "v_type" in type_dict[a.ctype]: # pass as vector
                    if type_dict[a.ctype]["v_type"] in ("Mat", "vector_Mat"): #pass as Mat or vector_Mat
                        jn_args.append  ( ArgInfo([ "__int64", "%s_mat.nativeObj" % a.name, "", [], "" ]) )
                        jni_args.append ( ArgInfo([ "__int64", "%s_mat_nativeObj" % a.name, "", [], "" ]) )
                        c_prologue.append( type_dict[a.ctype]["jni_var"] % {"n" : a.name} + ";" )
                        c_prologue.append( "Mat& %(n)s_mat = *((Mat*)%(n)s_mat_nativeObj)" % {"n" : a.name} + ";" )
                        if "I" in a.out or not a.out:
                            if type_dict[a.ctype]["v_type"] == "vector_Mat":
                                j_prologue.append( "List<Mat> %(n)s_tmplm = new ArrayList<Mat>((%(n)s != null) ? %(n)s.size() : 0);" % {"n" : a.name } )
                                j_prologue.append( "Mat %(n)s_mat = Converters.%(t)s_to_Mat(%(n)s, %(n)s_tmplm);" % {"n" : a.name, "t" : a.ctype} )
                            else:
                                if not type_dict[a.ctype]["j_type"].startswith("MatOf"):
                                    j_prologue.append( "Mat %(n)s_mat = Converters.%(t)s_to_Mat(%(n)s);" % {"n" : a.name, "t" : a.ctype} )
                                else:
                                    j_prologue.append( "Mat %s_mat = %s;" % (a.name, a.name) )
                            c_prologue.append( "Mat_to_%(t)s( %(n)s_mat, %(n)s );" % {"n" : a.name, "t" : a.ctype} )
                        else:
                            if not type_dict[a.ctype]["j_type"].startswith("MatOf"):
                                j_prologue.append( "Mat %s_mat = new Mat();" % a.name )
                            else:
                                j_prologue.append( "Mat %s_mat = %s;" % (a.name, a.name) )
                        if "O" in a.out:
                            if not type_dict[a.ctype]["j_type"].startswith("MatOf"):
                                j_epilogue.append("Converters.Mat_to_%(t)s(%(n)s_mat, %(n)s);" % {"t" : a.ctype, "n" : a.name})
                                j_epilogue.append( "%s_mat.release();" % a.name )
                            c_epilogue.append( "%(t)s_to_Mat( %(n)s, %(n)s_mat );" % {"n" : a.name, "t" : a.ctype} )
                    else: #pass as list
                        jn_args.append  ( ArgInfo([ a.ctype, a.name, "", [], "" ]) )
                        jni_args.append ( ArgInfo([ a.ctype, "%s_list" % a.name , "", [], "" ]) )
                        c_prologue.append(type_dict[a.ctype]["jni_var"] % {"n" : a.name} + ";")
                        if "I" in a.out or not a.out:
                            c_prologue.append("%(n)s = List_to_%(t)s(env, %(n)s_list);" % {"n" : a.name, "t" : a.ctype})
                        if "O" in a.out:
                            c_epilogue.append("Copy_%s_to_List(env,%s,%s_list);" % (a.ctype, a.name, a.name))
                else:
                    fields = type_dict[a.ctype].get("jn_args", ((a.ctype, ""),))
                    if "I" in a.out or not a.out or self.isWrapped(a.ctype): # input arg, pass by primitive fields
                        for f in fields:
                            jn_args.append ( ArgInfo([ f[0], a.name + f[1], "", [], "" ]) )
                            jni_args.append( ArgInfo([ f[0], a.name + normalize_field_name(f[1]), "", [], "" ]) )
                    if "O" in a.out and not self.isWrapped(a.ctype): # out arg, pass as double[]
                        jn_args.append ( ArgInfo([ "double[]", "%s_out" % a.name, "", [], "" ]) )
                        jni_args.append ( ArgInfo([ "double[]", "%s_out" % a.name, "", [], "" ]) )
                        j_prologue.append( "double[] %s_out = new double[%i];" % (a.name, len(fields)) )
                        c_epilogue.append(
                            "jdouble tmp_%(n)s[%(cnt)i] = {%(args)s}; env->SetDoubleArrayRegion(%(n)s_out, 0, %(cnt)i, tmp_%(n)s);" %
                            { "n" : a.name, "cnt" : len(fields), "args" : ", ".join(["(jdouble)" + a.name + f[1] for f in fields]) } )
                        if type_dict[a.ctype]["j_type"] in ('bool', 'int', 'long', 'float', 'double'):
                            j_epilogue.append('if(%(n)s!=null) %(n)s[0] = (%(t)s)%(n)s_out[0];' % {'n':a.name,'t':type_dict[a.ctype]["j_type"]})
                        else:
                            set_vals = []
                            i = 0
                            for f in fields:
                                set_vals.append( "%(n)s%(f)s = %(t)s%(n)s_out[%(i)i]" %
                                    {"n" : a.name, "t": ("("+type_dict[f[0]]["j_type"]+")", "")[f[0]=="double"], "f" : f[1], "i" : i}
                                )
                                i += 1
                            j_epilogue.append( "if("+a.name+"!=null){ " + "; ".join(set_vals) + "; } ")

            # calculate java method signature to check for uniqueness
            j_args = []
            for a in args:
                if not a.ctype: #hidden
                    continue
                jt = type_dict[a.ctype]["j_type"]
                if a.out and jt in ('bool', 'int', 'long', 'float', 'double'):
                    jt += '[]'
                j_args.append( jt + ' ' + a.name )
            j_signature = type_dict[fi.ctype]["j_type"] + " " + \
                fi.jname + "(" + ", ".join(j_args) + ")"
            logging.info("java: " + j_signature)

            if j_signature in j_signatures:
                if args:
                    args.pop()
                    continue
                else:
                    break

            # java part:
            # private java NATIVE method decl
            # e.g.
            # private static native void add_0(long src1, long src2, long dst, long mask, int dtype);
            jn_code.write( Template(
                "    private static native $type $name($args);\n").substitute(
                type = type_dict[fi.ctype].get("jn_type", "double[]"),
                name = fi.jname + '_' + str(suffix_counter),
                args = ", ".join(["%s %s" % (type_dict[a.ctype]["jn_type"], normalize_field_name(a.name)) for a in jn_args])
            ) )

            # java part:

            #java doc comment
            if fi.docstring:
                lines = fi.docstring.splitlines()
                returnTag = False
                javadocParams = []
                toWrite = []
                inCode = False
                for index, line in enumerate(lines):
                    p0 = line.find("@param")
                    if p0 != -1:
                        p0 += 7
                        p1 = line.find(' ', p0)
                        p1 = len(line) if p1 == -1 else p1
                        name = line[p0:p1]
                        javadocParams.append(name)
                        for arg in j_args:
                            if arg.endswith(" " + name):
                                toWrite.append(line);
                                break
                    else:
                        if "<code>" in line:
                            inCode = True
                        if "</code>" in line:
                            inCode = False
                        line = line.replace('@result ', '@return ')  # @result is valid in Doxygen, but invalid in Javadoc
                        if "@return " in line:
                            returnTag = True

                        if (not inCode and toWrite and not toWrite[-1] and
                                line and not line.startswith("\\") and not line.startswith("<ul>") and not line.startswith("@param")):
                                toWrite.append("<p>");

                        if index == len(lines) - 1:
                            for arg in j_args:
                                name = arg[arg.rfind(' ') + 1:]
                                if not name in javadocParams:
                                    toWrite.append(" * @param " + name + " automatically generated");
                            if type_dict[fi.ctype]["j_type"] and not returnTag and fi.ctype != "void":
                                toWrite.append(" * @return automatically generated");
                        toWrite.append(line);

                for line in toWrite:
                    j_code.write(" "*4 + line + "\n")
            if fi.annotation:
                j_code.write(" "*4 + "\n".join(fi.annotation) + "\n")

            # public java wrapper method impl (calling native one above)
            # e.g.
            # public static void add( Mat src1, Mat src2, Mat dst, Mat mask, int dtype )
            # { add_0( src1.nativeObj, src2.nativeObj, dst.nativeObj, mask.nativeObj, dtype );  }
            ret_type = fi.ctype
            if fi.ctype.endswith('*'):
                ret_type = ret_type[:-1]
            ret_val = type_dict[ret_type]["j_type"] + " retVal = " if j_epilogue else "return "
            tail = ""
            ret = "return retVal;" if j_epilogue else ""
            if "v_type" in type_dict[ret_type]:
                j_type = type_dict[ret_type]["j_type"]
                if type_dict[ret_type]["v_type"] in ("Mat", "vector_Mat"):
                    tail = ")"
                    if j_type.startswith('MatOf'):
                        ret_val += j_type + ".fromNativeAddr("
                    else:
                        ret_val = "Mat retValMat = new Mat("
                        j_prologue.append( j_type + ' retVal = new Array' + j_type+'();')
                        j_epilogue.append('Converters.Mat_to_' + ret_type + '(retValMat, retVal);')
                        ret = "return retVal;"
            elif ret_type.startswith("Ptr_"):
                constructor = type_dict[ret_type]["j_type"] + ".__fromPtr__("
                if j_epilogue:
                    ret_val = type_dict[fi.ctype]["j_type"] + " retVal = " + constructor
                else:
                    ret_val = "return " + constructor
                tail = ")"
            elif ret_type == "void":
                ret_val = ""
                ret = ""
            elif ret_type == "": # c-tor
                if fi.classname and ci.base:
                    ret_val = "super("
                    tail = ")"
                else:
                    ret_val = "nativeObj = "
                    tail = ";\n        long nativeObjCopy = nativeObj;\n        org.opencv.core.Mat.cleaner.register(this, () -> delete(nativeObjCopy))"
                ret = ""
            elif self.isWrapped(ret_type): # wrapped class
                constructor = self.getClass(ret_type).jname + "("
                if j_epilogue:
                    ret_val = type_dict[ret_type]["j_type"] + " retVal = new " + constructor
                else:
                    ret_val = "return new " + constructor
                tail = ")"
            elif "jn_type" not in type_dict[ret_type]:
                constructor = type_dict[ret_type]["j_type"] + "("
                if j_epilogue:
                    ret_val = type_dict[fi.ctype]["j_type"] + " retVal = new " + constructor
                else:
                    ret_val = "return new " + constructor
                tail = ")"

            static = "static"
            if fi.classname:
                static = fi.static

            j_code.write( Template(
"""    public $static$j_type$j_name($j_args) {$prologue
        $ret_val$jn_name($jn_args_call)$tail;$epilogue$ret
    }

"""
                ).substitute(
                    ret = "\n        " + ret if ret else "",
                    ret_val = ret_val,
                    tail = tail,
                    prologue = "\n        " + "\n        ".join(j_prologue) if j_prologue else "",
                    epilogue = "\n        " + "\n        ".join(j_epilogue) if j_epilogue else "",
                    static = static + " " if static else "",
                    j_type=type_dict[fi.ctype]["j_type"] + " " if type_dict[fi.ctype]["j_type"] else "",
                    j_name=fi.jname,
                    j_args=", ".join(j_args),
                    jn_name=fi.jname + '_' + str(suffix_counter),
                    jn_args_call=", ".join( [a.name for a in jn_args] ),
                )
            )


            # cpp part:
            # jni_func(..) { _retval_ = cv_func(..); return _retval_; }
            ret = "return _retval_;" if c_epilogue else ""
            default = "return 0;"
            if fi.ctype == "void":
                ret = ""
                default = ""
            elif not fi.ctype: # c-tor
                if self.isSmartClass(ci):
                    ret = "return (jlong)(new Ptr<%(ctype)s>(_retval_));" % { 'ctype': fi.fullClassCPP() }
                else:
                    ret = "return (jlong) _retval_;"
            elif "v_type" in type_dict[fi.ctype]: # c-tor
                if type_dict[fi.ctype]["v_type"] in ("Mat", "vector_Mat"):
                    ret = "return (jlong) _retval_;"
            elif fi.ctype in ['String', 'string']:
                ret = "return env->NewStringUTF(_retval_.c_str());"
                default = 'return env->NewStringUTF("");'
            elif self.isWrapped(fi.ctype): # wrapped class:
                ret = None
                if fi.ctype in self.classes:
                    ret_ci = self.classes[fi.ctype]
                    if self.isSmartClass(ret_ci):
                        ret = "return (jlong)(new Ptr<%(ctype)s>(new %(ctype)s(_retval_)));" % { 'ctype': ret_ci.fullNameCPP() }
                if ret is None:
                    ret = "return (jlong) new %s(_retval_);" % self.fullTypeNameCPP(fi.ctype)
            elif fi.ctype.startswith('Ptr_'):
                c_prologue.append("typedef Ptr<%s> %s;" % (self.fullTypeNameCPP(fi.ctype[4:]), fi.ctype))
                ret = "return (jlong)(new %(ctype)s(_retval_));" % { 'ctype':fi.ctype }
            elif self.isWrapped(ret_type): # pointer to wrapped class:
                ret = "return (jlong) _retval_;"
            elif type_dict[fi.ctype]["jni_type"] == "jdoubleArray":
                ret = "return _da_retval_;"

            # hack: replacing func call with property set/get
            name = fi.name
            if prop_name:
                if args:
                    name = prop_name + " = "
                else:
                    name = prop_name + ";//"

            cvname = fi.fullNameCPP()
            retval = self.fullTypeNameCPP(fi.ctype) + " _retval_ = " if ret else "return "
            if fi.ctype == "void":
                retval = ""
            elif fi.ctype == "String":
                retval = "cv::" + self.fullTypeNameCPP(fi.ctype) + " _retval_ = "
            elif fi.ctype == "string":
                retval = "std::string _retval_ = "
            elif "v_type" in type_dict[fi.ctype]: # vector is returned
                retval = type_dict[fi.ctype]['jni_var'] % {"n" : '_ret_val_vector_'} + " = "
                if type_dict[fi.ctype]["v_type"] in ("Mat", "vector_Mat"):
                    c_epilogue.append("Mat* _retval_ = new Mat();")
                    c_epilogue.append(fi.ctype+"_to_Mat(_ret_val_vector_, *_retval_);")
                else:
                    if ret:
                        c_epilogue.append("jobject _retval_ = " + fi.ctype + "_to_List(env, _ret_val_vector_);")
                    else:
                        c_epilogue.append("return " + fi.ctype + "_to_List(env, _ret_val_vector_);")
            if fi.classname:
                if not fi.ctype: # c-tor
                    if self.isSmartClass(ci):
                        retval = self.smartWrap(ci, fi.fullClassCPP()) + " _retval_ = "
                        cvname = "makePtr<" + fi.fullClassCPP() +">"
                    else:
                        retval = fi.fullClassCPP() + "* _retval_ = "
                        cvname = "new " + fi.fullClassCPP()
                elif fi.static:
                    cvname = fi.fullNameCPP()
                else:
                    cvname = ("me->" if  not self.isSmartClass(ci) else "(*me)->") + name
                    c_prologue.append(
                        "%(cls)s* me = (%(cls)s*) self; //TODO: check for NULL"
                            % { "cls" : self.smartWrap(ci, fi.fullClassCPP())}
                    )
            cvargs = []
            for a in args:
                if a.pointer:
                    jni_name = "&%(n)s"
                else:
                    jni_name = "%(n)s"
                    if not a.out and not "jni_var" in type_dict[a.ctype]:
                        # explicit cast to C type to avoid ambiguous call error on platforms (mingw)
                        # where jni types are different from native types (e.g. jint is not the same as int)
                        jni_name  = "(%s)%s" % (cast_to(a.ctype), jni_name)
                if not a.ctype: # hidden
                    jni_name = a.defval
                cvargs.append( type_dict[a.ctype].get("jni_name", jni_name) % {"n" : a.name})
                if "v_type" not in type_dict[a.ctype]:
                    if ("I" in a.out or not a.out or self.isWrapped(a.ctype)) and "jni_var" in type_dict[a.ctype]: # complex type
                        c_prologue.append(type_dict[a.ctype]["jni_var"] % {"n" : a.name} + ";")
                    if a.out and "I" not in a.out and not self.isWrapped(a.ctype) and a.ctype:
                        c_prologue.append("%s %s;" % (a.ctype, a.name))

            rtype = type_dict[fi.ctype].get("jni_type", "jdoubleArray")
            clazz = ci.jname
            cpp_code.write ( Template(
"""
JNIEXPORT $rtype JNICALL Java_org_opencv_${module}_${clazz}_$fname ($argst);

JNIEXPORT $rtype JNICALL Java_org_opencv_${module}_${clazz}_$fname
  ($args)
{
    ${namespace}
    static const char method_name[] = "$module::$fname()";
    try {
        LOGD("%s", method_name);$prologue
        $retval$cvname($cvargs);$epilogue$ret
    } catch(const std::exception &e) {
        throwJavaException(env, &e, method_name);
    } catch (...) {
        throwJavaException(env, 0, method_name);
    }$default
}


""" ).substitute(
        rtype = rtype,
        module = self.module.replace('_', '_1'),
        clazz = clazz.replace('_', '_1'),
        fname = (fi.jname + '_' + str(suffix_counter)).replace('_', '_1'),
        args  = ", ".join(["%s %s" % (type_dict[a.ctype].get("jni_type"), a.name) for a in jni_args]),
        argst = ", ".join([type_dict[a.ctype].get("jni_type") for a in jni_args]),
        prologue = "\n        " + "\n        ".join(c_prologue) if c_prologue else "",
        epilogue = "\n        " + "\n        ".join(c_epilogue) if c_epilogue else "",
        ret = "\n        " + ret if ret else "",
        cvname = cvname,
        cvargs = " " + ", ".join(cvargs) + " " if cvargs else "",
        default = "\n    " + default if default else "",
        retval = retval,
        namespace = ('using namespace ' + ci.namespace.replace('.', '::') + ';') if ci.namespace and ci.namespace != 'cv' else ''
    ) )

            # adding method signature to dictionary
            j_signatures.append(j_signature)

            # processing args with default values
            if args and args[-1].defval:
                args.pop()
            else:
                break



    def gen_class(self, ci):
        logging.info("%s", ci)
        # constants
        consts_map = {c.name: c for c in ci.private_consts}
        consts_map.update({c.name: c for c in ci.consts})
        def const_value(v):
            if v in consts_map:
                target = consts_map[v]
                assert target.value != v
                return const_value(target.value)
            return v
        if ci.private_consts:
            logging.info("%s", ci.private_consts)
            ci.j_code.write("""
    private static final int
            %s;\n\n""" % (",\n"+" "*12).join(["%s = %s" % (c.name, const_value(c.value)) for c in ci.private_consts])
            )
        if ci.consts:
            enumTypes = set(map(lambda c: c.enumType, ci.consts))
            grouped_consts = {enumType: [c for c in ci.consts if c.enumType == enumType] for enumType in enumTypes}
            for typeName in sorted(grouped_consts.keys(), key=lambda x: str(x) if x is not None else ""):
                consts = grouped_consts[typeName]
                logging.info("%s", consts)
                if typeName:
                    typeNameShort = typeName.rsplit(".", 1)[-1]
###################### Utilize Java enums ######################
#                    ci.j_code.write("""
#    public enum {1} {{
#        {0};
#
#        private final int id;
#        {1}(int id) {{ this.id = id; }}
#        {1}({1} _this) {{ this.id = _this.id; }}
#        public int getValue() {{ return id; }}
#    }}\n\n""".format((",\n"+" "*8).join(["%s(%s)" % (c.name, c.value) for c in consts]), typeName)
#                    )
################################################################
                    ci.j_code.write("""
    // C++: enum {1} ({2})
    public static final int
            {0};\n\n""".format((",\n"+" "*12).join(["%s = %s" % (c.name, c.value) for c in consts]), typeNameShort, typeName)
                    )
                else:
                    ci.j_code.write("""
    // C++: enum <unnamed>
    public static final int
            {0};\n\n""".format((",\n"+" "*12).join(["%s = %s" % (c.name, c.value) for c in consts]))
                    )
        # methods
        for fi in ci.getAllMethods():
            self.gen_func(ci, fi)
        # props
        for pi in ci.props:
            basename = ci.fullNameOrigin()
            # getter
            getter_name = basename + ".get_" + pi.name
            fi = FuncInfo( [getter_name, pi.ctype, [], []], self.namespaces ) # [ funcname, return_ctype, [modifiers], [args] ]
            self.gen_func(ci, fi, pi.name)
            if pi.rw:
                #setter
                setter_name = basename + ".set_" + pi.name
                fi = FuncInfo( [ setter_name, "void", [], [ [pi.ctype, pi.name, "", [], ""] ] ], self.namespaces)
                self.gen_func(ci, fi, pi.name)

        # manual ports
        if ci.name in ManualFuncs:
            for func in sorted(ManualFuncs[ci.name].keys()):
                logging.info("manual function: %s", func)
                fn = ManualFuncs[ci.name][func]
                ci.j_code.write("\n".join(fn["j_code"]))
                ci.jn_code.write("\n".join(fn["jn_code"]))
                ci.cpp_code.write("\n".join(fn["cpp_code"]))

        if ci.name != self.Module or ci.base:
            ci.jn_code.write(
"""
    // native support for deleting native object
    private static native void delete(long nativeObj);
""" )

            # native support for java finalize()
            ci.cpp_code.write(
"""
//
//  native support for deleting native object
//  static void %(cls)s::delete( __int64 self )
//
JNIEXPORT void JNICALL Java_org_opencv_%(module)s_%(j_cls)s_delete(JNIEnv*, jclass, jlong);

JNIEXPORT void JNICALL Java_org_opencv_%(module)s_%(j_cls)s_delete
  (JNIEnv*, jclass, jlong self)
{
    delete (%(cls)s*) self;
}

""" % {"module" : module.replace('_', '_1'), "cls" : self.smartWrap(ci, ci.fullNameCPP()), "j_cls" : ci.jname.replace('_', '_1')}
            )

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

        ci.smart = True  # smart class is not properly handled in case of base/derived classes
        return ci.smart

    def smartWrap(self, ci, fullname):
        '''
        Wraps fullname with Ptr<> if needed
        '''
        if self.isSmartClass(ci):
            return "Ptr<" + fullname + ">"
        return fullname

    def finalize(self, output_jni_path):
        list_file = os.path.join(output_jni_path, "opencv_jni.hpp")
        self.save(list_file, '\n'.join(['#include "%s"' % f for f in self.cpp_files]))


def copy_java_files(java_files_dir, java_base_path, default_package_path='org/opencv/'):
    global total_files, updated_files
    java_files = []
    re_filter = re.compile(r'^.+\.(java|kt)(.in)?$')
    for root, dirnames, filenames in os.walk(java_files_dir):
       java_files += [os.path.join(root, filename) for filename in filenames if re_filter.match(filename)]
    java_files = [f.replace('\\', '/') for f in java_files]

    re_package = re.compile(r'^package +(.+);')
    re_prefix = re.compile(r'^.+[\+/]([^\+]+).(java|kt)(.in)?$')
    for java_file in java_files:
        src = checkFileRemap(java_file)
        with open(src, 'r') as f:
            package_line = f.readline()
        m = re_prefix.match(java_file)
        target_fname = (m.group(1) + '.' + m.group(2)) if m else os.path.basename(java_file)
        m = re_package.match(package_line)
        if m:
            package = m.group(1)
            package_path = package.replace('.', '/')
        else:
            package_path = default_package_path
        #print(java_file, package_path, target_fname)
        dest = os.path.join(java_base_path, os.path.join(package_path, target_fname))
        assert dest[-3:] != '.in', dest + ' | ' + target_fname
        mkdir_p(os.path.dirname(dest))
        total_files += 1
        if (not os.path.exists(dest)) or (os.stat(src).st_mtime - os.stat(dest).st_mtime > 1):
            copyfile(src, dest)
            updated_files += 1

def sanitize_java_documentation_string(doc, type):
    if type == "class":
        doc = doc.replace("@param ", "")

    doc = re.sub(re.compile('\\\\f\\$(.*?)\\\\f\\$', re.DOTALL), '\\(' + r'\1' + '\\)', doc)
    doc = re.sub(re.compile('\\\\f\\[(.*?)\\\\f\\]', re.DOTALL), '\\(' + r'\1' + '\\)', doc)
    doc = re.sub(re.compile('\\\\f\\{(.*?)\\\\f\\}', re.DOTALL), '\\(' + r'\1' + '\\)', doc)

    doc = doc.replace("&", "&amp;") \
        .replace("\\<", "&lt;") \
        .replace("\\>", "&gt;") \
        .replace("<", "&lt;") \
        .replace(">", "&gt;") \
        .replace("$", "$$") \
        .replace("@anchor", "") \
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
        .replace("@returns", "@return") \
        .replace("@sa", "SEE:") \
        .replace("@see", "SEE:") \
        .replace("@snippet", "SNIPPET:") \
        .replace("@todo", "TODO:") \
        .replace("@warning ", "WARNING: ")

    doc = re.sub(re.compile('\\*\\*([^\\*]+?)\\*\\*', re.DOTALL), '<b>' + r'\1' + '</b>', doc)

    lines = doc.splitlines()

    lines = list(map(lambda x: x[x.find('*'):].strip() if x.lstrip().startswith("*") else x, lines))

    listInd = [];
    indexDiff = 0;
    for index, line in enumerate(lines[:]):
        if line.strip().startswith("-"):
            i = line.find("-")
            if not listInd or i > listInd[-1]:
                lines.insert(index + indexDiff, "  "*len(listInd) + "<ul>")
                indexDiff += 1
                listInd.append(i);
                lines.insert(index + indexDiff, "  "*len(listInd) + "<li>")
                indexDiff += 1
            elif i == listInd[-1]:
                lines.insert(index + indexDiff, "  "*len(listInd) + "</li>")
                indexDiff += 1
                lines.insert(index + indexDiff, "  "*len(listInd) + "<li>")
                indexDiff += 1
            elif len(listInd) > 1 and i == listInd[-2]:
                lines.insert(index + indexDiff, "  "*len(listInd) + "</li>")
                indexDiff += 1
                del listInd[-1]
                lines.insert(index + indexDiff, "  "*len(listInd) + "</ul>")
                indexDiff += 1
                lines.insert(index + indexDiff, "  "*len(listInd) + "<li>")
                indexDiff += 1
            else:
                lines.insert(index + indexDiff, "  "*len(listInd) + "</li>")
                indexDiff += 1
                del listInd[-1]
                lines.insert(index + indexDiff, "  "*len(listInd) + "</ul>")
                indexDiff += 1
                lines.insert(index + indexDiff, "  "*len(listInd) + "<ul>")
                indexDiff += 1
                listInd.append(i);
                lines.insert(index + indexDiff, "  "*len(listInd) + "<li>")
                indexDiff += 1
            lines[index + indexDiff] = lines[index + indexDiff][0:i] + lines[index + indexDiff][i + 1:]
        else:
            if listInd and (not line or line == "*" or line.strip().startswith("@note") or line.strip().startswith("@param")):
                lines.insert(index + indexDiff, "  "*len(listInd) + "</li>")
                indexDiff += 1
                del listInd[-1]
                lines.insert(index + indexDiff, "  "*len(listInd) + "</ul>")
                indexDiff += 1

    i = len(listInd) - 1
    for value in enumerate(listInd):
        lines.append("  "*i + "  </li>")
        lines.append("  "*i + "</ul>")
        i -= 1;

    lines = list(map(lambda x: "* " + x[1:].strip() if x.startswith("*") and x != "*" else x, lines))
    lines = list(map(lambda x: x if x.startswith("*") else "* " + x if x and x != "*" else "*", lines))

    lines = list(map(lambda x: x
        .replace("@note", "<b>Note:</b>")
    , lines))

    lines = list(map(lambda x: re.sub('@b ([\\w:]+?)\\b', '<b>' + r'\1' + '</b>', x), lines))
    lines = list(map(lambda x: re.sub('@c ([\\w:]+?)\\b', '<tt>' + r'\1' + '</tt>', x), lines))
    lines = list(map(lambda x: re.sub('`(.*?)`', "{@code " + r'\1' + '}', x), lines))
    lines = list(map(lambda x: re.sub('@p ([\\w:]+?)\\b', '{@code ' + r'\1' + '}', x), lines))

    hasValues = False
    for line in lines:
        if line != "*":
            hasValues = True
            break
    return "/**\n " + "\n ".join(lines) + "\n */" if hasValues else ""

if __name__ == "__main__":
    # initialize logger
    logging.basicConfig(filename='gen_java.log', format=None, filemode='w', level=logging.INFO)
    handler = logging.StreamHandler()
    handler.setLevel(os.environ.get('LOG_LEVEL', logging.WARNING))
    logging.getLogger().addHandler(handler)

    # parse command line parameters
    import argparse
    arg_parser = argparse.ArgumentParser(description='OpenCV Java Wrapper Generator')
    arg_parser.add_argument('-p', '--parser', required=True, help='OpenCV header parser')
    arg_parser.add_argument('-c', '--config', required=True, help='OpenCV modules config')

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
    FILES_REMAP = { os.path.realpath(os.path.join(ROOT_DIR, f['src'])): f['target'] for f in config['files_remap'] }
    logging.info("\nRemapped configured files (%d):\n%s", len(FILES_REMAP), pformat(FILES_REMAP))

    dstdir = "./gen"
    jni_path = os.path.join(dstdir, 'cpp'); mkdir_p(jni_path)
    java_base_path = os.path.join(dstdir, 'java'); mkdir_p(java_base_path)
    java_test_base_path = os.path.join(dstdir, 'test'); mkdir_p(java_test_base_path)

    for (subdir, target_subdir) in [('src/java', 'java'), ('android/java', None),
                                    ('android-21/java', None), ('android-24/java', None)]:
        if target_subdir is None:
            target_subdir = subdir
        java_files_dir = os.path.join(SCRIPT_DIR, subdir)
        if os.path.exists(java_files_dir):
            target_path = os.path.join(dstdir, target_subdir); mkdir_p(target_path)
            copy_java_files(java_files_dir, target_path)

    # launch Java Wrapper generator
    generator = JavaWrapperGenerator()

    gen_dict_files = []

    print("JAVA: Processing OpenCV modules: %d" % len(config['modules']))
    for e in config['modules']:
        (module, module_location) = (e['name'], os.path.join(ROOT_DIR, e['location']))
        logging.info("\n=== MODULE: %s (%s) ===\n" % (module, module_location))

        java_path = os.path.join(java_base_path, 'org/opencv')
        mkdir_p(java_path)

        module_imports = []
        module_j_code = None
        module_jn_code = None
        srcfiles = []
        common_headers = []

        misc_location = os.path.join(module_location, 'misc/java')

        srcfiles_fname = os.path.join(misc_location, 'filelist')
        if os.path.exists(srcfiles_fname):
            with open(srcfiles_fname) as f:
                srcfiles = [os.path.join(module_location, str(l).strip()) for l in f.readlines() if str(l).strip()]
        else:
            re_bad = re.compile(r'(private|.inl.hpp$|_inl.hpp$|.details.hpp$|_winrt.hpp$|/cuda/|/legacy/)')
            # .h files before .hpp
            h_files = []
            hpp_files = []
            for root, dirnames, filenames in os.walk(os.path.join(module_location, 'include')):
               h_files += [os.path.join(root, filename) for filename in fnmatch.filter(filenames, '*.h')]
               hpp_files += [os.path.join(root, filename) for filename in fnmatch.filter(filenames, '*.hpp')]
            srcfiles = h_files + hpp_files
            srcfiles = [f for f in srcfiles if not re_bad.search(f.replace('\\', '/'))]
        logging.info("\nFiles (%d):\n%s", len(srcfiles), pformat(srcfiles))

        common_headers_fname = os.path.join(misc_location, 'filelist_common')
        if os.path.exists(common_headers_fname):
            with open(common_headers_fname) as f:
                common_headers = [os.path.join(module_location, str(l).strip()) for l in f.readlines() if str(l).strip()]
        logging.info("\nCommon headers (%d):\n%s", len(common_headers), pformat(common_headers))

        gendict_fname = os.path.join(misc_location, 'gen_dict.json')
        if os.path.exists(gendict_fname):
            with open(gendict_fname) as f:
                gen_type_dict = json.load(f)
            class_ignore_list += gen_type_dict.get("class_ignore_list", [])
            const_ignore_list += gen_type_dict.get("const_ignore_list", [])
            const_private_list += gen_type_dict.get("const_private_list", [])
            missing_consts.update(gen_type_dict.get("missing_consts", {}))
            type_dict.update(gen_type_dict.get("type_dict", {}))
            ManualFuncs.update(gen_type_dict.get("ManualFuncs", {}))
            func_arg_fix.update(gen_type_dict.get("func_arg_fix", {}))
            namespaces_dict.update(gen_type_dict.get("namespaces_dict", {}))
            if 'module_j_code' in gen_type_dict:
                module_j_code = read_contents(checkFileRemap(os.path.join(misc_location, gen_type_dict['module_j_code'])))
            if 'module_jn_code' in gen_type_dict:
                module_jn_code = read_contents(checkFileRemap(os.path.join(misc_location, gen_type_dict['module_jn_code'])))
            module_imports += gen_type_dict.get("module_imports", [])

        java_files_dir = os.path.join(misc_location, 'src/java')
        if os.path.exists(java_files_dir):
            copy_java_files(java_files_dir, java_base_path, 'org/opencv/' + module)

        java_test_files_dir = os.path.join(misc_location, 'test')
        if os.path.exists(java_test_files_dir):
            copy_java_files(java_test_files_dir, java_test_base_path, 'org/opencv/test/' + module)

        if len(srcfiles) > 0:
            generator.gen(srcfiles, module, dstdir, jni_path, java_path, common_headers)
        else:
            logging.info("No generated code for module: %s", module)
    generator.finalize(jni_path)

    print('Generated files: %d (updated %d)' % (total_files, updated_files))
