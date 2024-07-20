# distutils: libraries = flint
# distutils: depends = flint/ca_ext.h

################################################################################
# This file is auto-generated by the script
#   SAGE_ROOT/src/sage_setup/autogen/flint_autogen.py.
# From the commit 3e2c3a3e091106a25ca9c6fba28e02f2cbcd654a
# Do not modify by hand! Fix and rerun the script instead.
################################################################################

from libc.stdio cimport FILE
from sage.libs.gmp.types cimport *
from sage.libs.mpfr.types cimport *
from sage.libs.flint.types cimport *

cdef extern from "flint_wrap.h":
    void ca_ext_init_qqbar(ca_ext_t res, const qqbar_t x, ca_ctx_t ctx) noexcept
    void ca_ext_init_const(ca_ext_t res, calcium_func_code func, ca_ctx_t ctx) noexcept
    void ca_ext_init_fx(ca_ext_t res, calcium_func_code func, const ca_t x, ca_ctx_t ctx) noexcept
    void ca_ext_init_fxy(ca_ext_t res, calcium_func_code func, const ca_t x, const ca_t y, ca_ctx_t ctx) noexcept
    void ca_ext_init_fxn(ca_ext_t res, calcium_func_code func, ca_srcptr x, slong nargs, ca_ctx_t ctx) noexcept
    void ca_ext_init_set(ca_ext_t res, const ca_ext_t x, ca_ctx_t ctx) noexcept
    void ca_ext_clear(ca_ext_t res, ca_ctx_t ctx) noexcept
    slong ca_ext_nargs(const ca_ext_t x, ca_ctx_t ctx) noexcept
    void ca_ext_get_arg(ca_t res, const ca_ext_t x, slong i, ca_ctx_t ctx) noexcept
    ulong ca_ext_hash(const ca_ext_t x, ca_ctx_t ctx) noexcept
    bint ca_ext_equal_repr(const ca_ext_t x, const ca_ext_t y, ca_ctx_t ctx) noexcept
    int ca_ext_cmp_repr(const ca_ext_t x, const ca_ext_t y, ca_ctx_t ctx) noexcept
    void ca_ext_print(const ca_ext_t x, ca_ctx_t ctx) noexcept
    void ca_ext_get_acb_raw(acb_t res, ca_ext_t x, slong prec, ca_ctx_t ctx) noexcept
    void ca_ext_cache_init(ca_ext_cache_t cache, ca_ctx_t ctx) noexcept
    void ca_ext_cache_clear(ca_ext_cache_t cache, ca_ctx_t ctx) noexcept
    ca_ext_ptr ca_ext_cache_insert(ca_ext_cache_t cache, const ca_ext_t x, ca_ctx_t ctx) noexcept
