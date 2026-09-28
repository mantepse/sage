r"""
Symmetric functions of `r`-coloured cycles

This module provides the power-sum algebra `\Lambda(r)` of
`r`-coloured symmetric functions in the sense of Henderson
[Henderson2004]_.  The case `r = 2` is the *hyperoctahedral* case.

The corresponding species live in
:mod:`sage.rings.species_hyperoctahedral`; the combinatorial model of
the canonical permutation representation of the wreath product
`W(r,n) = C_r \wr S_n` lives in
:mod:`sage.groups.perm_gps.hyperoctahedral_group`.

This module currently provides the *power-sum algebra* `\Lambda(r)`
only.  Following Henderson, `\Lambda(r)` is the polynomial algebra

.. MATH::

    \Lambda(r) = \mathbb{C}[p_i(\zeta) \mid i \in \mathbb{Z}^+, \zeta \in \mu_r]

graded by `\deg(p_i(\zeta)) = i`.  Rather than representing `\zeta` by
an actual root of unity, we label the `r` possible values of `\zeta` by
the integers `0, 1, \ldots, r-1`.

The algebra `\Lambda(r)` is identified with the tensor product
`\Lambda(1)^{\otimes r}` of `r` copies of the ordinary power-sum
algebra, the `j`-th tensor factor carrying the power sums
`p_i(\zeta^j)`.  Basis elements are indexed by
:class:`~sage.combinat.partition_tuple.PartitionTuple`\ s, so that

.. MATH::

    p_{\lambda^{(0)}, \ldots, \lambda^{(r-1)}}
    = \prod_{j=0}^{r-1} \prod_i p_i(\zeta^j)^{m_i(\lambda^{(j)})}.

EXAMPLES::

    sage: from sage.rings.sf_hyperoctahedral import HyperoctahedralSymmetricFunctions
    sage: H = HyperoctahedralSymmetricFunctions(QQ, 2)
    sage: H
    Hyperoctahedral symmetric functions of level 2 over Rational Field
    sage: H.power_sum(2, 0) * H.power_sum(1, 1)
    p_2(ζ^0)*p_1(ζ^1)

The two plethysms of Henderson are available through the public method
:meth:`~sage.rings.sf_hyperoctahedral.HyperoctahedralSymmetricFunctions.Element.plethysm`::

    sage: p = SymmetricFunctions(QQ).p()
    sage: p[2].plethysm(H.power_sum(3, 1))
    p_6(ζ^1)
    sage: H.power_sum(2, 1).plethysm(p[3])
    p_6(ζ^1)

The two generator rules are (with `p_i` an ordinary power sum and
`p_j(\zeta^a)` a generator of `\Lambda(r)`)::

    sage: H = HyperoctahedralSymmetricFunctions(QQ, 3)
    sage: p = SymmetricFunctions(QQ).p()
    sage: p[2].plethysm(H.power_sum(3, 1))     # p_2 o p_3(z^1) = p_6(z^1)
    p_6(ζ^1)
    sage: H.power_sum(2, 0).plethysm(p[3])     # p_2(z^0) o p_3 = p_6(z^0)
    p_6(ζ^0)
    sage: H.power_sum(2, 1).plethysm(p[3])     # p_2(z^1) o p_3 = p_6(z^3) = p_6(z^0)
    p_6(ζ^0)
    sage: H.power_sum(2, 1).plethysm(p[2])     # p_2(z^1) o p_2 = p_4(z^2)
    p_4(ζ^2)

Both plethysms are linear in the left argument, and the second one is
additive in the right argument for a single generator::

    sage: f = p[2] + 3*p[1, 1]
    sage: g = p[3] - p[1]
    sage: (f + g).plethysm(H.power_sum(3, 2)) == f.plethysm(H.power_sum(3, 2)) + g.plethysm(H.power_sum(3, 2))
    True
    sage: H.power_sum(2, 0).plethysm(f + g) == H.power_sum(2, 0).plethysm(f) + H.power_sum(2, 0).plethysm(g)
    True
    sage: F = H.power_sum(2, 1) + H.power_sum(1, 0)*H.power_sum(1, 1)
    sage: (F + H.power_sum(3, 0)).plethysm(f) == F.plethysm(f) + H.power_sum(3, 0).plethysm(f)
    True

The plethysms are associative whenever both sides are defined::

    sage: h = H.power_sum(2, 1) + H.power_sum(1, 0)*H.power_sum(1, 1)
    sage: f.plethysm(g).plethysm(h) == f.plethysm(g.plethysm(h))
    True
    sage: F.plethysm(g).plethysm(f) == F.plethysm(g.plethysm(f))
    True

REFERENCES:

.. [Henderson2004] Anthony Henderson.
   *Representations of wreath products on cohomology of De Concini-Procesi
   compactifications*.
   International Mathematics Research Notices 2004, no. 20, 981-1021.

AUTHORS:

- Martin Rubey (2025): initial version
"""


# ****************************************************************************
#       Copyright (C) 2025 Martin Rubey <martin.rubey@tuwien.ac.at>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#                  https://www.gnu.org/licenses/
# ****************************************************************************

from sage.categories.graded_algebras_with_basis import GradedAlgebrasWithBasis
from sage.categories.tensor import tensor
from sage.combinat.free_module import CombinatorialFreeModule
from sage.combinat.partition_tuple import PartitionTuple, PartitionTuples_level
from sage.combinat.sf.sf import SymmetricFunctions
from sage.misc.cachefunc import cached_method
from sage.rings.integer_ring import ZZ


class HyperoctahedralSymmetricFunctions(CombinatorialFreeModule):
    r"""
    The power-sum algebra `\Lambda(r)` of `r`-coloured symmetric
    functions.

    INPUT:

    - ``base_ring`` -- a commutative ring which is a `\QQ`-algebra
    - ``r`` -- positive integer; the order of the cyclic group

    The basis is indexed by :class:`~sage.combinat.partition_tuple.PartitionTuple`\ s
    of level `r`, the partition tuple `(\lambda^{(0)}, \ldots, \lambda^{(r-1)})`
    corresponding to the power-sum monomial

    .. MATH::

        p_{\lambda^{(0)}, \ldots, \lambda^{(r-1)}}
        = \prod_{j=0}^{r-1} \prod_i p_i(\zeta^j)^{m_i(\lambda^{(j)})}.

    EXAMPLES::

        sage: from sage.rings.sf_hyperoctahedral import HyperoctahedralSymmetricFunctions
        sage: H = HyperoctahedralSymmetricFunctions(QQ, 2)
        sage: H.basis_keys()
        Partition tuples of level 2
        sage: tau = H.basis_keys()([[2, 1], [3]])
        sage: tau
        ([2, 1], [3])
        sage: H.monomial(tau)
        p_{2,1}(ζ^0)*p_3(ζ^1)
        sage: H.monomial(tau).degree()
        6

    We can also construct generators directly::

        sage: H.power_sum(4, 1)
        p_4(ζ^1)
        sage: H.power_sum(4, 3)  # the type is taken modulo r
        p_4(ζ^1)

    The product is the obvious commutative product::

        sage: H.power_sum(2, 0) * H.power_sum(2, 1) * H.power_sum(1, 1)
        p_2(ζ^0)*p_{2,1}(ζ^1)

    TESTS::

        sage: H = HyperoctahedralSymmetricFunctions(QQ, 2)
        sage: TestSuite(H).run()
        sage: HyperoctahedralSymmetricFunctions(QQ, 2) is H
        True
        sage: HyperoctahedralSymmetricFunctions(QQ, 1)
        Hyperoctahedral symmetric functions of level 1 over Rational Field
    """
    @staticmethod
    def __classcall__(cls, base_ring, r):
        r"""
        Normalize the arguments for unique representation.

        TESTS::

            sage: from sage.rings.sf_hyperoctahedral import HyperoctahedralSymmetricFunctions
            sage: HyperoctahedralSymmetricFunctions(QQ, 2) is HyperoctahedralSymmetricFunctions(QQ, ZZ(2))
            True
            sage: HyperoctahedralSymmetricFunctions(QQ, 0)
            Traceback (most recent call last):
            ...
            ValueError: r must be a positive integer
        """
        r = ZZ(r)
        if r < 1:
            raise ValueError("r must be a positive integer")
        return super().__classcall__(cls, base_ring, r)

    def __init__(self, base_ring, r):
        r"""
        Initialize the power-sum algebra `\Lambda(r)`.

        TESTS::

            sage: from sage.rings.sf_hyperoctahedral import HyperoctahedralSymmetricFunctions
            sage: HyperoctahedralSymmetricFunctions(QQ, 3)._r
            3
        """
        self._r = ZZ(r)
        category = GradedAlgebrasWithBasis(base_ring).Commutative()
        # PartitionTuples(1) collapses to Partitions; use the level-1
        # parent directly so that keys are always partition tuples.
        indices = PartitionTuples_level(self._r)
        CombinatorialFreeModule.__init__(self, base_ring,
                                         basis_keys=indices,
                                         category=category,
                                         prefix='p', bracket=False)

    def _repr_(self):
        r"""
        Return a string representation of ``self``.

        EXAMPLES::

            sage: from sage.rings.sf_hyperoctahedral import HyperoctahedralSymmetricFunctions
            sage: HyperoctahedralSymmetricFunctions(QQ, 2)
            Hyperoctahedral symmetric functions of level 2 over Rational Field
        """
        return (f"Hyperoctahedral symmetric functions of level {self._r} "
                f"over {self.base_ring()}")

    def level(self):
        r"""
        Return the level `r` of ``self``.

        EXAMPLES::

            sage: from sage.rings.sf_hyperoctahedral import HyperoctahedralSymmetricFunctions
            sage: HyperoctahedralSymmetricFunctions(QQ, 3).level()
            3
        """
        return self._r

    def power(self):
        r"""
        Return the power-sum realization of `\Lambda(r)`.

        Since the power-sum basis is currently the only basis, this
        returns ``self``.

        EXAMPLES::

            sage: from sage.rings.sf_hyperoctahedral import HyperoctahedralSymmetricFunctions
            sage: H = HyperoctahedralSymmetricFunctions(QQ, 2)
            sage: H.power() is H
            True
        """
        return self

    def basis_keys(self):
        r"""
        Return the index set of the power-sum basis.

        EXAMPLES::

            sage: from sage.rings.sf_hyperoctahedral import HyperoctahedralSymmetricFunctions
            sage: HyperoctahedralSymmetricFunctions(QQ, 2).basis_keys()
            Partition tuples of level 2
        """
        return self._indices

    def power_sum(self, i, j):
        r"""
        Return the generator `p_i(\zeta^j)`.

        INPUT:

        - ``i`` -- positive integer; the degree
        - ``j`` -- integer; the colour, taken modulo `r`

        EXAMPLES::

            sage: from sage.rings.sf_hyperoctahedral import HyperoctahedralSymmetricFunctions
            sage: H = HyperoctahedralSymmetricFunctions(QQ, 3)
            sage: H.power_sum(2, 1)
            p_2(ζ^1)
            sage: H.power_sum(2, 4) == H.power_sum(2, 1)
            True
        """
        i = ZZ(i)
        if i < 1:
            raise ValueError("i must be a positive integer")
        j = ZZ(j) % self._r
        parts = [[] for _ in range(self._r)]
        parts[j] = [i]
        return self.monomial(self._indices(parts))

    # Alias expected by the general power-sum conventions.
    gen = power_sum

    def _repr_term(self, tau):
        r"""
        Return a string representation of the basis element ``tau``.

        EXAMPLES::

            sage: from sage.rings.sf_hyperoctahedral import HyperoctahedralSymmetricFunctions
            sage: H = HyperoctahedralSymmetricFunctions(QQ, 2)
            sage: H.power_sum(2, 0)
            p_2(ζ^0)
            sage: H.monomial(H.basis_keys()([[2, 1], [3]]))
            p_{2,1}(ζ^0)*p_3(ζ^1)
            sage: H.one()
            1
        """
        factors = []
        for j in range(self._r):
            lam = tau[j]
            if not lam:
                continue
            if len(lam) == 1:
                factors.append(f"p_{lam[0]}(ζ^{j})")
            else:
                factors.append("p_{" + ",".join(str(k) for k in lam) + f"}}(ζ^{j})")
        return "*".join(factors) if factors else "1"

    def one_basis(self):
        r"""
        Return the index of the multiplicative unit.

        EXAMPLES::

            sage: from sage.rings.sf_hyperoctahedral import HyperoctahedralSymmetricFunctions
            sage: HyperoctahedralSymmetricFunctions(QQ, 2).one_basis()
            ([], [])
        """
        return self._indices([[] for _ in range(self._r)])

    def degree_on_basis(self, tau):
        r"""
        Return the degree of the basis element ``tau``.

        EXAMPLES::

            sage: from sage.rings.sf_hyperoctahedral import HyperoctahedralSymmetricFunctions
            sage: H = HyperoctahedralSymmetricFunctions(QQ, 2)
            sage: H.degree_on_basis(H.basis_keys()([[2, 1], [3]]))
            6
        """
        return tau.size()

    def product_on_basis(self, mu, nu):
        r"""
        Return the product of the two basis elements ``mu`` and ``nu``.

        The partition tuple indexing a power-sum monomial records the
        multiplicities of the generators, so multiplication adds these
        multiplicities componentwise.

        EXAMPLES::

            sage: from sage.rings.sf_hyperoctahedral import HyperoctahedralSymmetricFunctions
            sage: H = HyperoctahedralSymmetricFunctions(QQ, 2)
            sage: K = H.basis_keys()
            sage: H.product_on_basis(K([[2], [1]]), K([[1], [3]]))
            p_{2,1}(ζ^0)*p_{3,1}(ζ^1)
        """
        parts = [sorted(list(mu[j]) + list(nu[j]), reverse=True)
                 for j in range(self._r)]
        return self.monomial(self._indices(parts))

    @cached_method
    def _tensor(self):
        r"""
        Return the tensor product realizing `\Lambda(r) \cong \Lambda(1)^{\otimes r}`.

        EXAMPLES::

            sage: from sage.rings.sf_hyperoctahedral import HyperoctahedralSymmetricFunctions
            sage: H = HyperoctahedralSymmetricFunctions(QQ, 2)
            sage: H._tensor()
            Symmetric Functions over Rational Field in the powersum basis # Symmetric Functions over Rational Field in the powersum basis
        """
        p = SymmetricFunctions(self.base_ring()).power()
        return tensor([p] * self._r)

    def _to_tensor(self, g):
        r"""
        Convert the element ``g`` of ``self`` to the tensor realization.

        EXAMPLES::

            sage: from sage.rings.sf_hyperoctahedral import HyperoctahedralSymmetricFunctions
            sage: H = HyperoctahedralSymmetricFunctions(QQ, 2)
            sage: H._to_tensor(H.power_sum(2, 0) + H.power_sum(1, 1))
            p[] # p[1] + p[2] # p[]
        """
        T = self._tensor()
        return T._from_dict({tuple(tau): c for tau, c in g},
                            remove_zeros=False)

    def _from_tensor(self, x):
        r"""
        Convert the tensor element ``x`` to an element of ``self``.

        EXAMPLES::

            sage: from sage.rings.sf_hyperoctahedral import HyperoctahedralSymmetricFunctions
            sage: H = HyperoctahedralSymmetricFunctions(QQ, 2)
            sage: T = H._tensor()
            sage: H._from_tensor(T._from_dict({((2,), (1,)): QQ(3)}, remove_zeros=False))
            3*p_2(ζ^0)*p_1(ζ^1)
        """
        return self._from_dict({self._indices(list(key)): c
                                for key, c in x},
                               remove_zeros=False)

    def _plethysm_type1(self, f, g):
        r"""
        Return the plethysm `f \circ g`, where `f` is ordinary.

        This is Henderson's first plethysm
        `\Lambda(1) \times \Lambda(r) \to \Lambda(r)`, defined by

        .. MATH::

            p_i \circ p_j(\zeta^a) = p_{ij}(\zeta^a).

        It is computed by identifying `\Lambda(r)` with
        `\Lambda(1)^{\otimes r}` and using Sage's plethysm of ordinary
        symmetric functions on each tensor factor.

        INPUT:

        - ``f`` -- an ordinary symmetric function
        - ``g`` -- an element of ``self``

        EXAMPLES::

            sage: from sage.rings.sf_hyperoctahedral import HyperoctahedralSymmetricFunctions
            sage: H = HyperoctahedralSymmetricFunctions(QQ, 2)
            sage: p = SymmetricFunctions(QQ).p()
            sage: H._plethysm_type1(p[2], H.power_sum(3, 1))
            p_6(ζ^1)
            sage: H._plethysm_type1(p[2], H.power_sum(2, 0)*H.power_sum(3, 1))
            p_4(ζ^0)*p_6(ζ^1)
        """
        p = f.parent().realization_of().power()
        return self._from_tensor(p(f)(self._to_tensor(g)))

    def _plethysm_type2(self, f, g):
        r"""
        Return the plethysm `f \circ g`, where ``g`` is ordinary.

        This is Henderson's second plethysm
        `\Lambda(r) \times \Lambda(1) \to \Lambda(r)`, defined by

        .. MATH::

            p_i(\zeta^a) \circ p_j = p_{ij}(\zeta^{aj}).

        INPUT:

        - ``f`` -- an element of ``self``
        - ``g`` -- an ordinary symmetric function

        EXAMPLES::

            sage: from sage.rings.sf_hyperoctahedral import HyperoctahedralSymmetricFunctions
            sage: H = HyperoctahedralSymmetricFunctions(QQ, 3)
            sage: p = SymmetricFunctions(QQ).p()
            sage: H._plethysm_type2(H.power_sum(2, 1), p[2])
            p_4(ζ^2)
            sage: H._plethysm_type2(H.power_sum(2, 1), p[2] + p[1, 1])
            p_4(ζ^2) + p_{2,2}(ζ^1)
        """
        p = g.parent().realization_of().power()
        g = p(g)
        result = self.zero()
        for nu, c in f:
            term = self.one()
            for a in range(self._r):
                for k in nu[a]:
                    term *= self._type2_factor(a, k, g)
            result += c * term
        return result

    def _type2_factor(self, a, k, g):
        r"""
        Return `p_k(\zeta^a) \circ g` for an ordinary power sum ``g``.

        INPUT:

        - ``a`` -- the colour of the left power sum
        - ``k`` -- the degree of the left power sum
        - ``g`` -- an ordinary symmetric function in the power-sum basis

        EXAMPLES::

            sage: from sage.rings.sf_hyperoctahedral import HyperoctahedralSymmetricFunctions
            sage: H = HyperoctahedralSymmetricFunctions(QQ, 2)
            sage: p = SymmetricFunctions(QQ).p()
            sage: H._type2_factor(1, 2, p[3])
            p_6(ζ^1)
            sage: H._type2_factor(1, 2, p[2])
            p_4(ζ^0)
        """
        result = self.zero()
        for lam, d in g:
            parts = [[] for _ in range(self._r)]
            for j in lam:
                parts[(a * j) % self._r].append(k * j)
            result += d * self.monomial(self._indices(parts))
        return result

    class Element(CombinatorialFreeModule.Element):
        r"""
        An element of `\Lambda(r)`.
        """
        def plethysm(self, x):
            r"""
            Return the plethysm of ``self`` with ``x``.

            The dispatch follows Henderson: if ``x`` is an ordinary
            symmetric function, the second plethysm
            `\Lambda(r) \times \Lambda(1) \to \Lambda(r)` is used.  If
            ``x`` is an element of `\Lambda(r)`, the `r`-by-`r`
            plethysm is not implemented yet.

            EXAMPLES::

                sage: from sage.rings.sf_hyperoctahedral import HyperoctahedralSymmetricFunctions
                sage: H = HyperoctahedralSymmetricFunctions(QQ, 3)
                sage: p = SymmetricFunctions(QQ).p()
                sage: H.power_sum(2, 1).plethysm(p[2])
                p_4(ζ^2)
                sage: H.power_sum(2, 1).plethysm(p[2] + p[1, 1])
                p_4(ζ^2) + p_{2,2}(ζ^1)

            The `r`-by-`r` plethysm is not implemented yet::

                sage: H.power_sum(2, 1).plethysm(H.power_sum(3, 0))
                Traceback (most recent call last):
                ...
                NotImplementedError: plethysm of two r-species is not implemented yet
            """
            if isinstance(x, HyperoctahedralSymmetricFunctions.Element):
                raise NotImplementedError("plethysm of two r-species is not implemented yet")
            return self.parent()._plethysm_type2(self, x)
