from .all__sagemath_objects import *

from .sage_eval import sage_eval, sageobj

from .sage_input import sage_input

from .banner import version

lazy_import('sage.misc.banner', 'banner', deprecation=34259)

lazy_import('sage.misc.sagedoc', ['browse_sage_doc',
        'search_src', 'search_def', 'search_doc',
        'tutorial', 'reference', 'manual', 'developer',
        'constructions', 'help'])

lazy_import('pydoc', 'help', 'python_help')

from .explain_pickle import explain_pickle, unpickle_newobj, unpickle_build, unpickle_instantiate, unpickle_persistent, unpickle_extension, unpickle_appends

lazy_import('sage.misc.trace', 'trace', deprecation=34259)

lazy_import('sage.misc.profiler', 'Profiler', deprecation=34259)

from .dev_tools import import_statements

lazy_import('sage.misc.dev_tools', 'runsnake', deprecation=34259)

from .edit_module import edit

lazy_import('sage.misc.edit_module', 'set_edit_template', deprecation=34259)

lazy_import('sage.misc.pager', 'pager')


lazy_import("sage.misc.cython", "cython_lambda")
lazy_import("sage.misc.cython", "cython_compile", "cython")
lazy_import('sage.misc.inline_fortran', 'fortran')

lazy_import('sage.misc.package', ('installed_packages', 'is_package_installed',
                                  'standard_packages', 'optional_packages',
                                  'experimental_packages', 'package_versions'))

lazy_import('sage.misc.benchmark', 'benchmark', deprecation=34259)
