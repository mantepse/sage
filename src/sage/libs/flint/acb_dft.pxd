# distutils: libraries = flint
# distutils: depends = flint/acb_dft.h

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
    void acb_dft(acb_ptr w, acb_srcptr v, slong n, slong prec) noexcept
    void acb_dft_inverse(acb_ptr w, acb_srcptr v, slong n, slong prec) noexcept
    void acb_dft_precomp_init(acb_dft_pre_t pre, slong len, slong prec) noexcept
    void acb_dft_precomp_clear(acb_dft_pre_t pre) noexcept
    void acb_dft_precomp(acb_ptr w, acb_srcptr v, const acb_dft_pre_t pre, slong prec) noexcept
    void acb_dft_inverse_precomp(acb_ptr w, acb_srcptr v, const acb_dft_pre_t pre, slong prec) noexcept
    void acb_dirichlet_dft_prod(acb_ptr w, acb_srcptr v, slong * cyc, slong num, slong prec) noexcept
    void acb_dft_prod_init(acb_dft_prod_t t, slong * cyc, slong num, slong prec) noexcept
    void acb_dft_prod_clear(acb_dft_prod_t t) noexcept
    void acb_dirichlet_dft_prod_precomp(acb_ptr w, acb_srcptr v, const acb_dft_prod_t prod, slong prec) noexcept
    void acb_dft_convol_naive(acb_ptr w, acb_srcptr f, acb_srcptr g, slong len, slong prec) noexcept
    void acb_dft_convol_rad2(acb_ptr w, acb_srcptr f, acb_srcptr g, slong len, slong prec) noexcept
    void acb_dft_convol(acb_ptr w, acb_srcptr f, acb_srcptr g, slong len, slong prec) noexcept
    void acb_dft_naive(acb_ptr w, acb_srcptr v, slong n, slong prec) noexcept
    void acb_dft_naive_init(acb_dft_naive_t t, slong len, slong prec) noexcept
    void acb_dft_naive_clear(acb_dft_naive_t t) noexcept
    void acb_dft_naive_precomp(acb_ptr w, acb_srcptr v, const acb_dft_naive_t t, slong prec) noexcept
    void acb_dft_crt(acb_ptr w, acb_srcptr v, slong n, slong prec) noexcept
    void acb_dft_crt_init(acb_dft_crt_t t, slong len, slong prec) noexcept
    void acb_dft_crt_clear(acb_dft_crt_t t) noexcept
    void acb_dft_crt_precomp(acb_ptr w, acb_srcptr v, const acb_dft_crt_t t, slong prec) noexcept
    void acb_dft_cyc(acb_ptr w, acb_srcptr v, slong n, slong prec) noexcept
    void acb_dft_cyc_init(acb_dft_cyc_t t, slong len, slong prec) noexcept
    void acb_dft_cyc_clear(acb_dft_cyc_t t) noexcept
    void acb_dft_cyc_precomp(acb_ptr w, acb_srcptr v, const acb_dft_cyc_t t, slong prec) noexcept
    void acb_dft_rad2(acb_ptr w, acb_srcptr v, int e, slong prec) noexcept
    void acb_dft_inverse_rad2(acb_ptr w, acb_srcptr v, int e, slong prec) noexcept
    void acb_dft_rad2_init(acb_dft_rad2_t t, int e, slong prec) noexcept
    void acb_dft_rad2_clear(acb_dft_rad2_t t) noexcept
    void acb_dft_rad2_precomp(acb_ptr w, acb_srcptr v, const acb_dft_rad2_t t, slong prec) noexcept
    void acb_dft_bluestein(acb_ptr w, acb_srcptr v, slong n, slong prec) noexcept
    void acb_dft_bluestein_init(acb_dft_bluestein_t t, slong len, slong prec) noexcept
    void acb_dft_bluestein_clear(acb_dft_bluestein_t t) noexcept
    void acb_dft_bluestein_precomp(acb_ptr w, acb_srcptr v, const acb_dft_bluestein_t t, slong prec) noexcept
