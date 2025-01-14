from typing import Sequence, Generator, Tuple, Optional, Union
import weakref
import abc
from itertools import chain

from .node import ASTNode, ASTNodeType


class TypeResolutionError(Exception):
    pass


class TypeNode(abc.ABC):
    """This class and its derivatives used for construction parts of AST that
    otherwise can't be constructed from the information provided by header
    parser, because this information is either not available at that moment of
    time or not available at all:
        - There is no possible way to derive correspondence between C++ type
          and its Python equivalent if it is not exposed from library
          e.g. `cv::Rect`.
        - There is no information about types visibility (see `ASTNodeTypeNode`).
    """
    compatible_to_runtime_usage = False
    """Class-wide property that switches exported type names for several nodes.
    Example:
    >>> node = OptionalTypeNode(ASTNodeTypeNode("Size"))
    >>> node.typename  # TypeNode.compatible_to_runtime_usage == False
    "Size | None"
    >>> TypeNode.compatible_to_runtime_usage = True
    >>> node.typename
    "typing.Optional[Size]"
    """

    def __init__(self, ctype_name: str, required_modules: Tuple[str, ...] = ()) -> None:
        self.ctype_name = ctype_name
        self._required_modules = required_modules

    @abc.abstractproperty
    def typename(self) -> str:
        """Short name of the type node used that should be used in the same
        module (or a file) where type is defined.

        Returns:
            str: short name of the type node.
        """
        pass

    @property
    def full_typename(self) -> str:
        """Full name of the type node including full module name starting from
        the package.
        Example: 'cv2.Algorithm', 'cv2.gapi.ie.PyParams'.

        Returns:
            str: full name of the type node.
        """
        return self.typename

    @property
    def required_definition_imports(self) -> Generator[str, None, None]:
        """Generator filled with import statements required for type
        node definition (especially used by `AliasTypeNode`).

        Example:
        ```python
        # Alias defined in the `cv2.typing.__init__.pyi`
        Callback = typing.Callable[[cv2.GMat, float], None]

        # alias definition
        callback_alias = AliasTypeNode.callable_(
            'Callback',
            arg_types=(ASTNodeTypeNode('GMat'), PrimitiveTypeNode.float_())
        )

        # Required definition imports
        for required_import in callback_alias.required_definition_imports:
            print(required_import)
        # Outputs:
        # 'import typing'
        # 'import cv2'
        ```

        Yields:
            Generator[str, None, None]: generator filled with import statements
                required for type node definition.
        """
        yield from ()

    @property
    def required_usage_imports(self) -> Generator[str, None, None]:
        """Generator filled with import statements required for type node
        usage.

        Example:
        ```python
        # Alias defined in the `cv2.typing.__init__.pyi`
        Callback = typing.Callable[[cv2.GMat, float], None]

        # alias definition
        callback_alias = AliasTypeNode.callable_(
            'Callback',
            arg_types=(ASTNodeTypeNode('GMat'), PrimitiveTypeNode.float_())
        )

        # Required usage imports
        for required_import in callback_alias.required_usage_imports:
            print(required_import)
        # Outputs:
        # 'import cv2.typing'
        ```

        Yields:
            Generator[str, None, None]: generator filled with import statements
                required for type node definition.
        """
        yield from ()

    @property
    def required_modules(self) -> Tuple[str, ...]:
        return self._required_modules

    @property
    def is_resolved(self) -> bool:
        return True

    def relative_typename(self, module_full_export_name: str) -> str:
        """Type name relative to the provided module.

        Args:
            module_full_export_name (str): Full export name of the module to
                get relative name to.

        Returns:
            str: If module name of the type node doesn't match `module`, then
                returns class scopes + `self.typename`, otherwise
                `self.full_typename`.
        """
        return self.full_typename

    def resolve(self, root: ASTNode) -> None:
        """Resolves all references to AST nodes using a top-down search
        for nodes with corresponding export names. See `_resolve_symbol` for
        more details.

        Args:
            root (ASTNode): Node pointing to the root of a subtree in AST
                representing search scope of the symbol.
                Most of the symbols don't have full paths in their names, so
                scopes should be examined in bottom-up manner starting
                with narrowest one.

        Raises:
            TypeResolutionError: if at least 1 reference to AST node can't
                be resolved in the subtree pointed by the root.
        """
        pass


class NoneTypeNode(TypeNode):
    """Type node representing a None (or `void` in C++) type.
    """
    @property
    def typename(self) -> str:
        return "None"


class AnyTypeNode(TypeNode):
    """Type node representing any type (most of the time it means unknown).
    """
    @property
    def typename(self) -> str:
        return "_typing.Any"

    @property
    def required_usage_imports(self) -> Generator[str, None, None]:
        yield "import typing as _typing"


class PrimitiveTypeNode(TypeNode):
    """Type node representing a primitive built-in types e.g. int, float, str.
    """
    def __init__(self, ctype_name: str,
                 typename: Optional[str] = None,
                 required_modules: Tuple[str, ...] = ()) -> None:
        super().__init__(ctype_name, required_modules)
        self._typename = typename if typename is not None else ctype_name

    @property
    def typename(self) -> str:
        return self._typename

    @classmethod
    def int_(cls, ctype_name: Optional[str] = None,
             required_modules: Tuple[str, ...] = ()):
        if ctype_name is None:
            ctype_name = "int"
        return PrimitiveTypeNode(ctype_name, typename="int", required_modules=required_modules)

    @classmethod
    def float_(cls, ctype_name: Optional[str] = None,
               required_modules: Tuple[str, ...] = ()):
        if ctype_name is None:
            ctype_name = "float"
        return PrimitiveTypeNode(ctype_name, typename="float", required_modules=required_modules)

    @classmethod
    def bool_(cls, ctype_name: Optional[str] = None,
              required_modules: Tuple[str, ...] = ()):
        if ctype_name is None:
            ctype_name = "bool"
        return PrimitiveTypeNode(ctype_name, typename="bool", required_modules=required_modules)

    @classmethod
    def str_(cls, ctype_name: Optional[str] = None,
             required_modules: Tuple[str, ...] = ()):
        if ctype_name is None:
            ctype_name = "string"
        return PrimitiveTypeNode(ctype_name, "str", required_modules=required_modules)


class AliasRefTypeNode(TypeNode):
    """Type node representing an alias referencing another alias. Example:
    ```python
    Point2i = tuple[int, int]
    Point = Point2i
    ```
    During typing stubs generation procedure above code section might be defined
    as follows
    ```python
    AliasTypeNode.tuple_("Point2i",
                         items=(
                            PrimitiveTypeNode.int_(),
                            PrimitiveTypeNode.int_()
                         ))
    AliasTypeNode.ref_("Point", "Point2i")
    ```
    """
    def __init__(self, alias_ctype_name: str,
                 alias_export_name: Optional[str] = None,
                 required_modules: Tuple[str, ...] = ()):
        super().__init__(alias_ctype_name, required_modules)
        if alias_export_name is None:
            self.alias_export_name = alias_ctype_name
        else:
            self.alias_export_name = alias_export_name

    @property
    def typename(self) -> str:
        return self.alias_export_name

    @property
    def full_typename(self) -> str:
        return "cv2.typing." + self.typename


class AliasTypeNode(TypeNode):
    """Type node representing an alias to another type.
    Example:
    ```python
    Point2i = tuple[int, int]
    ```
    can be defined as
    ```python
    AliasTypeNode.tuple_("Point2i",
                         items=(
                            PrimitiveTypeNode.int_(),
                            PrimitiveTypeNode.int_()
                         ))
    ```
    Under the hood it is implemented as a container of another type node.
    """
    def __init__(self, ctype_name: str, value: TypeNode,
                 export_name: Optional[str] = None,
                 doc: Optional[str] = None,
                 required_modules: Tuple[str, ...] = ()) -> None:
        super().__init__(ctype_name, required_modules)
        self.value = value
        self._export_name = export_name
        self.doc = doc

    @property
    def typename(self) -> str:
        if self._export_name is not None:
            return self._export_name
        return self.ctype_name

    @property
    def full_typename(self) -> str:
        return "cv2.typing." + self.typename

    @property
    def required_definition_imports(self) -> Generator[str, None, None]:
        return self.value.required_usage_imports

    @property
    def required_usage_imports(self) -> Generator[str, None, None]:
        yield "import cv2.typing"

    @property
    def is_resolved(self) -> bool:
        return self.value.is_resolved

    def resolve(self, root: ASTNode):
        try:
            self.value.resolve(root)
        except TypeResolutionError as e:
            raise TypeResolutionError(
                'Failed to resolve alias "{}" exposed as "{}"'.format(
                    self.ctype_name, self.typename
                )
            ) from e

    @classmethod
    def int_(cls, ctype_name: str, export_name: Optional[str] = None,
             doc: Optional[str] = None, required_modules: Tuple[str, ...] = ()):
        return cls(ctype_name, PrimitiveTypeNode.int_(), export_name, doc, required_modules)

    @classmethod
    def float_(cls, ctype_name: str, export_name: Optional[str] = None,
               doc: Optional[str] = None, required_modules: Tuple[str, ...] = ()):
        return cls(ctype_name, PrimitiveTypeNode.float_(), export_name, doc, required_modules)

    @classmethod
    def array_ref_(cls, ctype_name: str, array_ref_name: str,
                   shape: Optional[Tuple[int, ...]],
                   dtype: Optional[str] = None,
                   export_name: Optional[str] = None,
                   doc: Optional[str] = None,
                   required_modules: Tuple[str, ...] = ()):
        """Create alias to array reference alias `array_ref_name`.

        This is required to preserve backward compatibility with Python < 3.9
        and NumPy 1.20, when NumPy module introduces generics support.

        Args:
            ctype_name (str): Name of the alias.
            array_ref_name (str): Name of the conditional array alias.
            shape (Optional[Tuple[int, ...]]): Array shape.
            dtype (Optional[str], optional): Array type.  Defaults to None.
            export_name (Optional[str], optional): Alias export name.
                Defaults to None.
            doc (Optional[str], optional): Documentation string for alias.
                Defaults to None.
        """
        if doc is None:
            doc = f"NDArray(shape={shape}, dtype={dtype})"
        else:
            doc += f". NDArray(shape={shape}, dtype={dtype})"
        return cls(ctype_name, AliasRefTypeNode(array_ref_name),
                   export_name, doc, required_modules)

    @classmethod
    def union_(cls, ctype_name: str, items: Tuple[TypeNode, ...],
               export_name: Optional[str] = None,
               doc: Optional[str] = None,
               required_modules: Tuple[str, ...] = ()):
        return cls(ctype_name, UnionTypeNode(ctype_name, items),
                   export_name, doc, required_modules)

    @classmethod
    def optional_(cls, ctype_name: str, item: TypeNode,
                  export_name: Optional[str] = None,
                  doc: Optional[str] = None,
                  required_modules: Tuple[str, ...] = ()):
        return cls(ctype_name, OptionalTypeNode(item), export_name, doc, required_modules)

    @classmethod
    def sequence_(cls, ctype_name: str, item: TypeNode,
                  export_name: Optional[str] = None,
                  doc: Optional[str] = None,
                  required_modules: Tuple[str, ...] = ()):
        return cls(ctype_name, SequenceTypeNode(ctype_name, item),
                   export_name, doc, required_modules)

    @classmethod
    def tuple_(cls, ctype_name: str, items: Tuple[TypeNode, ...],
               export_name: Optional[str] = None,
               doc: Optional[str] = None,
               required_modules: Tuple[str, ...] = ()):
        return cls(ctype_name, TupleTypeNode(ctype_name, items),
                   export_name, doc, required_modules)

    @classmethod
    def class_(cls, ctype_name: str, class_name: str,
               export_name: Optional[str] = None,
               doc: Optional[str] = None,
               required_modules: Tuple[str, ...] = ()):
        return cls(ctype_name, ASTNodeTypeNode(class_name),
                   export_name, doc, required_modules)

    @classmethod
    def callable_(cls, ctype_name: str,
                  arg_types: Union[TypeNode, Sequence[TypeNode]],
                  ret_type: TypeNode = NoneTypeNode("void"),
                  export_name: Optional[str] = None,
                  doc: Optional[str] = None,
                  required_modules: Tuple[str, ...] = ()):
        return cls(ctype_name,
                   CallableTypeNode(ctype_name, arg_types, ret_type),
                   export_name, doc, required_modules)

    @classmethod
    def ref_(cls, ctype_name: str, alias_ctype_name: str,
             alias_export_name: Optional[str] = None,
             export_name: Optional[str] = None,
             doc: Optional[str] = None,
             required_modules: Tuple[str, ...] = ()):
        return cls(ctype_name,
                   AliasRefTypeNode(alias_ctype_name, alias_export_name),
                   export_name, doc, required_modules)

    @classmethod
    def dict_(cls, ctype_name: str, key_type: TypeNode, value_type: TypeNode,
              export_name: Optional[str] = None, doc: Optional[str] = None,
              required_modules: Tuple[str, ...] = ()):
        return cls(ctype_name, DictTypeNode(ctype_name, key_type, value_type),
                   export_name, doc, required_modules)


class ConditionalAliasTypeNode(TypeNode):
    """Type node representing an alias protected by condition checked in runtime.
    For typing-related conditions, prefer using typing.TYPE_CHECKING. For a full explanation, see:
    https://github.com/opencv/opencv/pull/23927#discussion_r1256326835

    Example:
    ```python
    if typing.TYPE_CHECKING
        NumPyArray = numpy.ndarray[typing.Any, numpy.dtype[numpy.generic]]
    else:
        NumPyArray = numpy.ndarray
    ```
    is defined as follows:
    ```python

    ConditionalAliasTypeNode(
        "NumPyArray",
        'typing.TYPE_CHECKING',
        NDArrayTypeNode("NumPyArray"),
        NDArrayTypeNode("NumPyArray", use_numpy_generics=False),
        condition_required_imports=("import typing",)
    )
    ```
    """
    def __init__(self, ctype_name: str, condition: str,
                 positive_branch_type: TypeNode,
                 negative_branch_type: TypeNode,
                 export_name: Optional[str] = None,
                 condition_required_imports: Sequence[str] = ()) -> None:
        super().__init__(ctype_name)
        self.condition = condition
        self.positive_branch_type = positive_branch_type
        self.positive_branch_type.ctype_name = self.ctype_name
        self.negative_branch_type = negative_branch_type
        self.negative_branch_type.ctype_name = self.ctype_name
        self._export_name = export_name
        self._condition_required_imports = condition_required_imports

    @property
    def typename(self) -> str:
        if self._export_name is not None:
            return self._export_name
        return self.ctype_name

    @property
    def full_typename(self) -> str:
        return "cv2.typing." + self.typename

    @property
    def required_definition_imports(self) -> Generator[str, None, None]:
        yield from self.positive_branch_type.required_usage_imports
        yield from self.negative_branch_type.required_usage_imports
        yield from self._condition_required_imports

    @property
    def required_usage_imports(self) -> Generator[str, None, None]:
        yield "import cv2.typing"

    @property
    def required_modules(self) -> Tuple[str, ...]:
        return (*self.positive_branch_type.required_modules,
                *self.negative_branch_type.required_modules)

    @property
    def is_resolved(self) -> bool:
        return self.positive_branch_type.is_resolved \
                and self.negative_branch_type.is_resolved

    def resolve(self, root: ASTNode):
        try:
            self.positive_branch_type.resolve(root)
            self.negative_branch_type.resolve(root)
        except TypeResolutionError as e:
            raise TypeResolutionError(
                'Failed to resolve alias "{}" exposed as "{}"'.format(
                    self.ctype_name, self.typename
                )
            ) from e

    @classmethod
    def numpy_array_(cls, ctype_name: str, export_name: Optional[str] = None,
                     shape: Optional[Tuple[int, ...]] = None,
                     dtype: Optional[str] = None):
        """Type subscription is not possible in python 3.8 and older numpy versions."""
        return cls(
            ctype_name,
            "_typing.TYPE_CHECKING",
            NDArrayTypeNode(ctype_name, shape, dtype),
            NDArrayTypeNode(ctype_name, shape, dtype,
                            use_numpy_generics=False),
            condition_required_imports=("import typing as _typing",)
        )


class NDArrayTypeNode(TypeNode):
    """Type node representing NumPy ndarray.
    """
    def __init__(self, ctype_name: str,
                 shape: Optional[Tuple[int, ...]] = None,
                 dtype: Optional[str] = None,
                 use_numpy_generics: bool = True) -> None:
        super().__init__(ctype_name)
        self.shape = shape
        self.dtype = dtype
        self._use_numpy_generics = use_numpy_generics

    @property
    def typename(self) -> str:
        if self._use_numpy_generics:
            # NOTE: Shape is not fully supported yet
            dtype = self.dtype if self.dtype is not None else "numpy.generic"
            return f"numpy.ndarray[_typing.Any, numpy.dtype[{dtype}]]"
        return "numpy.ndarray"

    @property
    def required_usage_imports(self) -> Generator[str, None, None]:
        yield "import numpy"
        # if self.shape is None:
        yield "import typing as _typing"


class ASTNodeTypeNode(TypeNode):
    """Type node representing a lazy ASTNode corresponding to type of
    function argument or its return type or type of class property.
    Introduced laziness nature resolves the types visibility issue - all types
    should be known during function declaration to select an appropriate node
    from the AST. Such knowledge leads to evaluation of all preprocessor
    directives (`#include` particularly) for each processed header and might be
    too expensive and error prone.
    """
    def __init__(self, ctype_name: str, typename: Optional[str] = None,
                 module_name: Optional[str] = None,
                 required_modules: Tuple[str, ...] = ()) -> None:
        super().__init__(ctype_name, required_modules)
        self._typename = typename if typename is not None else ctype_name
        self._module_name = module_name
        self._ast_node: Optional[weakref.ProxyType[ASTNode]] = None

    @property
    def ast_node(self):
        return self._ast_node

    @property
    def typename(self) -> str:
        if self._ast_node is None:
            return self._typename
        typename = self._ast_node.export_name
        if self._ast_node.node_type is not ASTNodeType.Enumeration:
            return typename
        # NOTE: Special handling for enums
        parent = self._ast_node.parent
        while parent.node_type is ASTNodeType.Class:
            typename = parent.export_name + "_" + typename
            parent = parent.parent
        return typename

    @property
    def full_typename(self) -> str:
        if self._ast_node is not None:
            if self._ast_node.node_type is not ASTNodeType.Enumeration:
                return self._ast_node.full_export_name
            # NOTE: enumerations are exported to module scope
            typename = self._ast_node.export_name
            parent = self._ast_node.parent
            while parent.node_type is ASTNodeType.Class:
                typename = parent.export_name + "_" + typename
                parent = parent.parent
            return parent.full_export_name + "." + typename
        if self._module_name is not None:
            return self._module_name + "." + self._typename
        return self._typename

    @property
    def required_usage_imports(self) -> Generator[str, None, None]:
        if self._module_name is None:
            assert self._ast_node is not None, \
                "Can't find a module for class '{}' exported as '{}'".format(
                    self.ctype_name, self.typename,
                )
            module = self._ast_node.parent
            while module.node_type is not ASTNodeType.Namespace:
                module = module.parent
            yield "import " + module.full_export_name
        else:
            yield "import " + self._module_name

    @property
    def is_resolved(self) -> bool:
        return self._ast_node is not None or self._module_name is not None

    def resolve(self, root: ASTNode):
        if self.is_resolved:
            return

        node = _resolve_symbol(root, self.typename)
        if node is None:
            raise TypeResolutionError('Failed to resolve "{}" exposed as "{}"'.format(
                self.ctype_name, self.typename
            ))
        self._ast_node = weakref.proxy(node)

    def relative_typename(self, module: str) -> str:
        assert self._ast_node is not None or self._module_name is not None, \
            "'{}' exported as '{}' is not resolved yet".format(self.ctype_name,
                                                               self.typename)
        if self._module_name is None:
            type_module = self._ast_node.parent  # type: ignore
            while type_module.node_type is not ASTNodeType.Namespace:
                type_module = type_module.parent
            module_name = type_module.full_export_name
        else:
            module_name = self._module_name
        if module_name != module:
            return self.full_typename
        return self.full_typename[len(module_name) + 1:]


class AggregatedTypeNode(TypeNode):
    """Base type node for type nodes representing an aggregation of another
    type nodes e.g. tuple, sequence or callable."""
    def __init__(self, ctype_name: str, items: Sequence[TypeNode],
                 required_modules: Tuple[str, ...] = ()) -> None:
        super().__init__(ctype_name, required_modules)
        self.items = list(items)

    @property
    def is_resolved(self) -> bool:
        return all(item.is_resolved for item in self.items)

    @property
    def required_modules(self) -> Tuple[str, ...]:
        return (*chain.from_iterable(item.required_modules for item in self.items),
                *self._required_modules)

    def resolve(self, root: ASTNode) -> None:
        errors = []
        for item in filter(lambda item: not item.is_resolved, self):
            try:
                item.resolve(root)
            except TypeResolutionError as e:
                errors.append(str(e))
        if len(errors) > 0:
            raise TypeResolutionError(
                'Failed to resolve one of "{}" items. Errors: {}'.format(
                    self.full_typename, errors
                )
            )

    def __iter__(self):
        return iter(self.items)

    def __len__(self) -> int:
        return len(self.items)

    @property
    def required_definition_imports(self) -> Generator[str, None, None]:
        for item in self:
            yield from item.required_definition_imports

    @property
    def required_usage_imports(self) -> Generator[str, None, None]:
        for item in self:
            yield from item.required_usage_imports


class ContainerTypeNode(AggregatedTypeNode):
    """Base type node for all type nodes representing a container type.
    """
    @property
    def typename(self) -> str:
        return self.type_format.format(self.types_separator.join(
            item.typename for item in self
        ))

    @property
    def full_typename(self) -> str:
        return self.type_format.format(self.types_separator.join(
            item.full_typename for item in self
        ))

    def relative_typename(self, module: str) -> str:
        return self.type_format.format(self.types_separator.join(
            item.relative_typename(module) for item in self
        ))

    @property
    def required_definition_imports(self) -> Generator[str, None, None]:
        yield "import typing as _typing"
        yield from super().required_definition_imports

    @property
    def required_usage_imports(self) -> Generator[str, None, None]:
        if TypeNode.compatible_to_runtime_usage:
            yield "import typing as _typing"
        yield from super().required_usage_imports

    @abc.abstractproperty
    def type_format(self) -> str:
        pass

    @abc.abstractproperty
    def types_separator(self) -> str:
        pass


class SequenceTypeNode(ContainerTypeNode):
    """Type node representing a homogeneous collection of elements with
    possible unknown length.
    """
    def __init__(self, ctype_name: str, item: TypeNode,
                 required_modules: Tuple[str, ...] = ()) -> None:
        super().__init__(ctype_name, (item, ), required_modules)

    @property
    def type_format(self) -> str:
        return "_typing.Sequence[{}]"

    @property
    def types_separator(self) -> str:
        return ", "


class TupleTypeNode(ContainerTypeNode):
    """Type node representing possibly heterogeneous collection of types with
    possibly unspecified length.
    """
    @property
    def type_format(self) -> str:
        if TypeNode.compatible_to_runtime_usage:
            return "_typing.Tuple[{}]"
        return "tuple[{}]"

    @property
    def types_separator(self) -> str:
        return ", "


class UnionTypeNode(ContainerTypeNode):
    """Type node representing type that can be one of the predefined set of types.
    """
    @property
    def type_format(self) -> str:
        if TypeNode.compatible_to_runtime_usage:
            return "_typing.Union[{}]"
        return "{}"

    @property
    def types_separator(self) -> str:
        if TypeNode.compatible_to_runtime_usage:
            return ", "
        return " | "


class OptionalTypeNode(ContainerTypeNode):
    """Type node representing optional type which is effectively is a union
    of value type node and None.
    """
    def __init__(self, value: TypeNode,
                 required_modules: Tuple[str, ...] = ()) -> None:
        super().__init__(value.ctype_name, (value,), required_modules)

    @property
    def type_format(self) -> str:
        if TypeNode.compatible_to_runtime_usage:
            return "_typing.Optional[{}]"
        return "{} | None"

    @property
    def types_separator(self) -> str:
        return ", "


class DictTypeNode(ContainerTypeNode):
    """Type node representing a homogeneous key-value mapping.
    """
    def __init__(self, ctype_name: str, key_type: TypeNode,
                 value_type: TypeNode,
                 required_modules: Tuple[str, ...] = ()) -> None:
        super().__init__(ctype_name, (key_type, value_type), required_modules)

    @property
    def key_type(self) -> TypeNode:
        return self.items[0]

    @property
    def value_type(self) -> TypeNode:
        return self.items[1]

    @property
    def type_format(self) -> str:
        if TypeNode.compatible_to_runtime_usage:
            return "_typing.Dict[{}]"
        return "dict[{}]"

    @property
    def types_separator(self) -> str:
        return ", "


class CallableTypeNode(AggregatedTypeNode):
    """Type node representing a callable type (most probably a function).

    ```python
    CallableTypeNode(
        'image_reading_callback',
        arg_types=(ASTNodeTypeNode('Image'), PrimitiveTypeNode.float_())
    )
    ```
    defines a callable type node representing a function with the same
    interface as the following
    ```python
    def image_reading_callback(image: Image, timestamp: float) -> None: ...
    ```
    """
    def __init__(self, ctype_name: str,
                 arg_types: Union[TypeNode, Sequence[TypeNode]],
                 ret_type: TypeNode = NoneTypeNode("void"),
                 required_modules: Tuple[str, ...] = ()) -> None:
        if isinstance(arg_types, TypeNode):
            super().__init__(ctype_name, (arg_types, ret_type), required_modules)
        else:
            super().__init__(ctype_name, (*arg_types, ret_type), required_modules)

    @property
    def arg_types(self) -> Sequence[TypeNode]:
        return self.items[:-1]

    @property
    def ret_type(self) -> TypeNode:
        return self.items[-1]

    @property
    def typename(self) -> str:
        return '_typing.Callable[[{}], {}]'.format(
            ', '.join(arg.typename for arg in self.arg_types),
            self.ret_type.typename
        )

    @property
    def full_typename(self) -> str:
        return '_typing.Callable[[{}], {}]'.format(
            ', '.join(arg.full_typename for arg in self.arg_types),
            self.ret_type.full_typename
        )

    def relative_typename(self, module: str) -> str:
        return '_typing.Callable[[{}], {}]'.format(
            ', '.join(arg.relative_typename(module) for arg in self.arg_types),
            self.ret_type.relative_typename(module)
        )

    @property
    def required_definition_imports(self) -> Generator[str, None, None]:
        yield "import typing as _typing"
        yield from super().required_definition_imports

    @property
    def required_usage_imports(self) -> Generator[str, None, None]:
        yield "import typing as _typing"
        yield from super().required_usage_imports


class ClassTypeNode(ContainerTypeNode):
    """Type node representing types themselves (refer to typing.Type)
    """
    def __init__(self, value: TypeNode,
                 required_modules: Tuple[str, ...] = ()) -> None:
        super().__init__(value.ctype_name, (value,), required_modules)

    @property
    def type_format(self) -> str:
        return "_typing.Type[{}]"

    @property
    def types_separator(self) -> str:
        return ", "


def _resolve_symbol(root: Optional[ASTNode], full_symbol_name: str) -> Optional[ASTNode]:
    """Searches for a symbol with the given full export name in the AST
    starting from the `root`.

    Args:
        root (Optional[ASTNode]): Root of the examining AST.
        full_symbol_name (str): Full export name of the symbol to find. Path
            components can be divided by '.' or '_'.

    Returns:
        Optional[ASTNode]: ASTNode with full export name equal to
            `full_symbol_name`, None otherwise.

    >>> root = NamespaceNode('cv')
    >>> cls = root.add_class('Algorithm').add_class('Params')
    >>> _resolve_symbol(root, 'cv.Algorithm.Params') == cls
    True

    >>> root = NamespaceNode('cv')
    >>> enum = root.add_namespace('detail').add_enumeration('AlgorithmType')
    >>> _resolve_symbol(root, 'cv_detail_AlgorithmType') == enum
    True

    >>> root = NamespaceNode('cv')
    >>> _resolve_symbol(root, 'cv.detail.Algorithm')
    None

    >>> root = NamespaceNode('cv')
    >>> enum = root.add_namespace('detail').add_enumeration('AlgorithmType')
    >>> _resolve_symbol(root, 'AlgorithmType')
    None
    """
    def search_down_symbol(scope: Optional[ASTNode],
                           scope_sep: str) -> Optional[ASTNode]:
        parts = full_symbol_name.split(scope_sep, maxsplit=1)
        while len(parts) == 2:
            # Try to find narrow scope
            scope = _resolve_symbol(scope, parts[0])
            if scope is None:
                return None
            # and resolve symbol in it
            node = _resolve_symbol(scope, parts[1])
            if node is not None:
                return node
            # symbol is not found, but narrowed scope is valid - diving further
            parts = parts[1].split(scope_sep, maxsplit=1)
        return None

    assert root is not None, \
        "Can't resolve symbol '{}' from NONE root".format(full_symbol_name)
    # Looking for exact symbol match
    for attr in filter(lambda attr: hasattr(root, attr),
                       ("namespaces", "classes", "enumerations")):
        nodes_dict = getattr(root, attr)
        node = nodes_dict.get(full_symbol_name, None)
        if node is not None:
            return node
    # Symbol is not found, looking for more fine-grained scope if possible
    for scope_sep in ("_", "."):
        node = search_down_symbol(root, scope_sep)
        if node is not None:
            return node
    return None
