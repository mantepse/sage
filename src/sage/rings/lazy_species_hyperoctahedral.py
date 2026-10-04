r"""
Lazy polynomial `r`-species

The graded algebra of polynomial `r`-species, described in
:mod:`sage.rings.species_hyperoctahedral`, is completed here in the same
way as the ring of polynomial species is completed in
:mod:`sage.rings.lazy_species`: the coefficient of degree `n` of a lazy
`r`-species is a finite linear combination of molecular `r`-species of
degree `n`, where the degree is the number of `C_r`-blocks.  Addition
and multiplication are performed coefficientwise.

In addition, the *type 1 substitution* of Henderson is available: an
ordinary lazy species `F`, possibly multisort, can be evaluated at lazy
`r`-species `G_1, \dots, G_k` of the same sort, where `k` is the number
of sorts of `F`.  The result is again a lazy `r`-species.

EXAMPLES:

We construct the ring of lazy `2`-species over the rationals.  The
generator `X` corresponds to the trivial subgroup of the wreath product
`W(r, 1)`, the molecular species `X^\circ` corresponds to its full
cyclic stabilizer::

    sage: from sage.rings.lazy_species_hyperoctahedral import LazyHyperoctahedralSpecies
    sage: from sage.groups.perm_gps.hyperoctahedral_group import _wreath_group
    sage: L.<X> = LazyHyperoctahedralSpecies(QQ, 2)
    sage: Xo = L(_wreath_group(2, 1))
    sage: X, Xo
    (X, X°)

Addition and multiplication are coefficientwise::

    sage: F = X + Xo
    sage: F
    (X+X°)
    sage: F**2
    (X^2+2*X*X°+X°^2)

The generating series of a lazy `r`-species `F` is
`\sum_{n \geq 0} |F[n]_r| x^n/(n! r^n)`, where `F[n]_r` denotes the
`F`-structures on the canonical free `W(r,n)`-set with `n` blocks.  For
molecular `r`-species, this amounts to replacing each molecular species
`M` with coefficient `c` by `c / |H_M|`, where `H_M` is the stabilizer
group of `M`::

    sage: F.generating_series()
    3/2*X + O(X^7)
    sage: Xo.generating_series()
    1/2*X + O(X^7)

The isotype generating series counts isomorphism classes of structures::

    sage: F.isotype_generating_series()
    2*X + O(X^7)

Substituting the molecular species `X^\circ` into the species of sets
provides the species `E^{(r)}` of `r`-sets, i.e., the unique
`r`-species with precisely one structure in each degree::

    sage: from sage.rings.lazy_species import LazyCombinatorialSpecies
    sage: E = LazyCombinatorialSpecies(QQ, "Z").Sets()
    sage: Er = E(Xo); Er
    1 + X° + E_2(X°) + E_3(X°) + E_4(X°) + E_5(X°) + E_6(X°) + O^7
    sage: Er.generating_series()
    1 + 1/2*X + 1/8*X^2 + 1/48*X^3 + 1/384*X^4 + 1/3840*X^5 + 1/46080*X^6 + O(X^7)
    sage: exp(Xo.generating_series())[:7] == Er.generating_series()[:7]
    True

The type 1 substitution distributes over addition and scalar
multiplication, and is associative::

    sage: from sage.groups.perm_gps.permgroup_named import SymmetricGroup
    sage: L1 = LazyCombinatorialSpecies(QQ, "Z")
    sage: E2 = L1(SymmetricGroup(2))
    sage: E2(X + Xo)[2]
    E_2(X) + X*X° + E_2(X°)
    sage: E2(2*X)[2]
    2*E_2(X) + X^2
    sage: E2(Xo)[2]
    E_2(X°)
    sage: E3 = L1(SymmetricGroup(3))
    sage: E2(E3(F))[4] == E2(E3)(F)[4]
    True

The generating series of a type 1 substitution is the substitution of
the generating series [Henderson2004, Section 4]::

    sage: E(X + Xo).generating_series()
    1 + 3/2*X + 9/8*X^2 + 9/16*X^3 + 27/128*X^4 + 81/1280*X^5 + 81/5120*X^6 + O(X^7)
    sage: C = L1.Cycles()
    sage: C(Xo).generating_series()
    1/2*X + 1/8*X^2 + 1/24*X^3 + 1/64*X^4 + 1/160*X^5 + 1/384*X^6 + 1/896*X^7 + O(X^8)
    sage: E(C(Xo)).generating_series()
    1 + 1/2*X + 1/4*X^2 + 1/8*X^3 + 1/16*X^4 + 1/32*X^5 + 1/64*X^6 + O(X^7)

Multisort `r`-species are supported::

    sage: from sage.rings.species_hyperoctahedral import PolynomialHyperoctahedralSpecies
    sage: P2 = PolynomialHyperoctahedralSpecies(QQ, 2, "X2, Y2")
    sage: L2.<X2, Y2> = LazyHyperoctahedralSpecies(QQ, 2)
    sage: Yo = L2(P2(_wreath_group(2, 1), {1: [1, 2]}))
    sage: E(X2 + Yo)[:3]
    [1, X2 + Y2°, E_2(X2) + X2*Y2° + E_2(Y2°)]

For `r = 1` the type 1 substitution reduces to the usual substitution
of species::

    sage: L1r.<X1> = LazyHyperoctahedralSpecies(QQ, 1)
    sage: from sage.rings.species_hyperoctahedral import PolynomialHyperoctahedralSpecies
    sage: P1 = PolynomialHyperoctahedralSpecies(QQ, 1, "X1")
    sage: E2(X1)[2] == L1r(P1(SymmetricGroup(2)))[2]
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
#  Distributed under the terms of the GNU General Public License (GPL)
#  as published by the Free Software Foundation; either version 2 of
#  the License, or (at your option) any later version.
#                  https://www.gnu.org/licenses/
# ****************************************************************************

import itertools
from collections import defaultdict

from sage.misc.lazy_list import lazy_list
from sage.rings.integer_ring import ZZ
from sage.rings.rational_field import QQ
from sage.rings.lazy_series import LazyCompletionGradedAlgebraElement
from sage.rings.lazy_series_ring import (LazyCompletionGradedAlgebra,
                                         LazyPowerSeriesRing)
from sage.rings.lazy_species import weighted_vector_compositions
from sage.rings.species import PolynomialSpecies
from sage.rings.species_hyperoctahedral import PolynomialHyperoctahedralSpecies
from sage.data_structures.stream import (Stream_exact,
                                         Stream_function,
                                         Stream_zero)
from sage.structure.element import get_coercion_model, parent


class LazyHyperoctahedralSpeciesElement(LazyCompletionGradedAlgebraElement):
    r"""
    A lazy `r`-species.

    EXAMPLES::

        sage: from sage.rings.lazy_species_hyperoctahedral import LazyHyperoctahedralSpecies
        sage: from sage.groups.perm_gps.hyperoctahedral_group import _wreath_group
        sage: L.<X> = LazyHyperoctahedralSpecies(QQ, 2)
        sage: Xo = L(_wreath_group(2, 1))
        sage: F = X + Xo
        sage: F
        (X+X°)
        sage: F**2
        (X^2+2*X*X°+X°^2)
    """

    def generating_series(self):
        r"""
        Return the generating series of ``self``.

        The coefficient of `x^n` is `|F[n]_r|/(n! r^n)`, where `F[n]_r`
        denotes the set of structures of ``self`` on the canonical free
        `W(r,n)`-set with `n` blocks.

        For multisort species the variables correspond to the sorts,
        the coefficient of `x_1^{n_1} \cdots x_k^{n_k}` is
        `|F[n_1, \ldots, n_k]_r| / |W(r; n_1, \ldots, n_k)|`.

        EXAMPLES::

            sage: from sage.rings.lazy_species_hyperoctahedral import LazyHyperoctahedralSpecies
            sage: from sage.groups.perm_gps.hyperoctahedral_group import _wreath_group
            sage: L.<X> = LazyHyperoctahedralSpecies(QQ, 2)
            sage: Xo = L(_wreath_group(2, 1))
            sage: (X + Xo).generating_series()
            3/2*X + O(X^7)

            sage: E = LazyCombinatorialSpecies(QQ, "Z").Sets()
            sage: E(Xo).generating_series()
            1 + 1/2*X + 1/8*X^2 + 1/48*X^3 + 1/384*X^4 + 1/3840*X^5 + 1/46080*X^6 + O(X^7)

            sage: from sage.rings.species_hyperoctahedral import PolynomialHyperoctahedralSpecies
            sage: P2 = PolynomialHyperoctahedralSpecies(QQ, 2, "X, Y")
            sage: L2.<X, Y> = LazyHyperoctahedralSpecies(QQ, 2)
            sage: Yo = L2(P2(_wreath_group(2, 1), {1: [1, 2]}))
            sage: E(X + Yo).generating_series()[:3]
            [1, X + 1/2*Y, 1/2*X^2 + 1/2*X*Y + 1/8*Y^2]
        """
        P = self.parent()
        L = LazyPowerSeriesRing(P.base_ring().fraction_field(),
                                P._laurent_poly_ring._indices._indices.variable_names())
        if P._arity == 1:
            def coefficient(n):
                return sum(c / M.permutation_group()[0].cardinality()
                           for M, c in self[n].monomial_coefficients().items())
        else:
            def coefficient(n):
                return sum(c / M.permutation_group()[0].cardinality()
                           * P.base_ring().prod(v ** d for v, d in zip(L.gens(), M.grade()))
                           for M, c in self[n].monomial_coefficients().items())
        return L(coefficient)

    def isotype_generating_series(self):
        r"""
        Return the isotype generating series of ``self``.

        The coefficient of `x^n` is the number of isomorphism classes of
        structures of ``self`` on the canonical free `W(r,n)`-set with
        `n` blocks.

        EXAMPLES::

            sage: from sage.rings.lazy_species_hyperoctahedral import LazyHyperoctahedralSpecies
            sage: from sage.groups.perm_gps.hyperoctahedral_group import _wreath_group
            sage: L.<X> = LazyHyperoctahedralSpecies(QQ, 2)
            sage: Xo = L(_wreath_group(2, 1))
            sage: (X + Xo).isotype_generating_series()
            2*X + O(X^7)

            sage: E = LazyCombinatorialSpecies(QQ, "Z").Sets()
            sage: E(Xo).isotype_generating_series()
            1 + X + X^2 + X^3 + X^4 + X^5 + X^6 + O(X^7)
        """
        P = self.parent()
        L = LazyPowerSeriesRing(P.base_ring().fraction_field(),
                                P._laurent_poly_ring._indices._indices.variable_names())
        if P._arity == 1:
            def coefficient(n):
                return sum(c for M, c in self[n].monomial_coefficients().items())
        else:
            def coefficient(n):
                return sum(c * P.base_ring().prod(v ** d for v, d in zip(L.gens(), M.grade()))
                           for M, c in self[n].monomial_coefficients().items())
        return L(coefficient)

    def structures(self, *labels):
        r"""
        Iterate over the structures on the given free `C_r`-set of labels.

        The labels are given as one list of `C_r`-orbits per sort, each
        orbit an iterable of exactly `r` labels in cyclic phase order;
        the degree is the number of orbits.  See
        :meth:`~sage.rings.species_hyperoctahedral.PolynomialHyperoctahedralSpecies.Element.structures`.

        EXAMPLES::

            sage: from sage.rings.lazy_species_hyperoctahedral import LazyHyperoctahedralSpecies
            sage: from sage.groups.perm_gps.hyperoctahedral_group import _wreath_group
            sage: L = LazyHyperoctahedralSpecies(QQ, 2)
            sage: Xo = L(_wreath_group(2, 1))
            sage: X = L(_wreath_group(2, 1).subgroup([]))
            sage: list(Xo.structures([('a', 'b')]))
            [(X°, (('a', 'b'),))]
            sage: sorted((X + Xo).structures([('a', 'b')]), key=str)
            [(X, (('a', 'b'),)), (X, (('b', 'a'),)), (X°, (('a', 'b'),))]

            The number of structures agrees with `r^n n!` times the
            coefficient of the generating series::

            sage: E = LazyCombinatorialSpecies(QQ, "Z").Sets()
            sage: sorted(E(Xo).structures([('a', 'b'), ('c', 'd'), ('e', 'f')]))
            [(E_3(X°), (('a', 'b', 'c', 'd', 'e', 'f'),))]
            sage: E(Xo).generating_series()[3]
            1/48
        """
        yield from self[sum(map(len, labels))].structures(*labels)

    def _test_structures(self, tester=None, max_size=4, **options):
        r"""
        Check that structures and generating series are consistent.

        We check all structures with less than ``max_size`` orbits.

        TESTS::

            sage: from sage.rings.lazy_species_hyperoctahedral import LazyHyperoctahedralSpecies
            sage: from sage.groups.perm_gps.hyperoctahedral_group import _wreath_group, _wreath_young_subgroup
            sage: L = LazyHyperoctahedralSpecies(QQ, 2)
            sage: Xo = L(_wreath_group(2, 1))
            sage: X = L(_wreath_group(2, 1).subgroup([]))
            sage: (X + Xo)._test_structures()
            sage: (X*Xo)._test_structures()

            Multisort species are checked for every vector of orbit
            counts::

            sage: from sage.rings.species_hyperoctahedral import MolecularHyperoctahedralSpecies, AtomicHyperoctahedralSpecies
            sage: LXY = LazyHyperoctahedralSpecies(QQ, 2, "X, Y")
            sage: MXY = MolecularHyperoctahedralSpecies(2, "X, Y")
            sage: AXY = AtomicHyperoctahedralSpecies(2, "X, Y")
            sage: W = _wreath_young_subgroup(2, [1, 1])
            sage: d = AXY(W.subgroup([[(1, 2), (3, 4)]]), {0: [1, 2], 1: [3, 4]})
            sage: LXY(MXY({d: 1}))._test_structures()
        """
        if tester is None:
            tester = self._tester(**options)
        P = self.parent()
        r = int(P._internal_poly_ring.base_ring()._r)
        for n in range(max_size):
            if P._arity == 1:
                labels = [[(i, j) for j in range(r)] for i in range(n)]
                s = list(self.structures(labels))
                tester.assertEqual(len(s), len(set(s)),
                                   f"structures for {labels} are {s}, which is not a set")
                coeff = self.generating_series()[n]
                tester.assertEqual(len(s) / ZZ(r)**n / ZZ(n).factorial(), coeff,
                                   f"the number of structures for {labels} is {len(s)}, "
                                   f"but the generating series gives {coeff}")
            else:
                from sage.combinat.integer_vector import IntegerVectors
                for shape in IntegerVectors(n, length=P._arity):
                    labels = [[[(i, j, sort) for j in range(r)]
                               for i in range(k)]
                              for sort, k in enumerate(shape)]
                    s = list(self.structures(*labels))
                    tester.assertEqual(len(s), len(set(s)),
                                       f"structures for {labels} are {s}, which is not a set")
                    coeff = self.generating_series()[n].coefficient(list(shape))
                    scale = ZZ.prod(ZZ(r)**k * ZZ(k).factorial() for k in shape)
                    tester.assertEqual(len(s) / scale, coeff,
                                       f"the number of structures for {labels} is {len(s)}, "
                                       f"but the generating series gives {coeff}")


class Type1CompositionSpeciesElement(LazyHyperoctahedralSpeciesElement):
    r"""
    The type 1 substitution of lazy `r`-species into an ordinary lazy
    species.
    """

    def __init__(self, left, *args):
        r"""
        Initialize the type 1 substitution of ``args`` into ``left``.

        INPUT:

        - ``left`` -- an ordinary lazy species with `k` sorts

        - ``args`` -- `k` lazy `r`-species, all with the same `r` and
          sorts

        TESTS::

            sage: from sage.rings.lazy_species_hyperoctahedral import LazyHyperoctahedralSpecies
            sage: from sage.rings.lazy_species import LazyCombinatorialSpecies
            sage: from sage.groups.perm_gps.hyperoctahedral_group import _wreath_group
            sage: from sage.groups.perm_gps.permgroup_named import SymmetricGroup
            sage: L.<X> = LazyHyperoctahedralSpecies(QQ, 2)
            sage: Xo = L(_wreath_group(2, 1))
            sage: L1 = LazyCombinatorialSpecies(QQ, "Z")
            sage: E2 = L1(SymmetricGroup(2))
            sage: F = E2(X + Xo)
            sage: TestSuite(F).run(skip=['_test_category', '_test_pickling'])

            sage: L1.zero()(Xo)
            0
            sage: E2(L.zero())
            0
            sage: (1 + E2)(L.zero())
            1
            sage: (1 + E2)(Xo)
            1 + E_2(X°) + O^7

        Substituting a constant series raises an error::

            sage: E2(1 + Xo)
            Traceback (most recent call last):
            ...
            ValueError: can only compose with a positive valuation series

        All substituted species must be lazy `r`-species with common
        `r` and sorts::

            sage: L2.<A, B> = LazyCombinatorialSpecies(QQ)
            sage: L3.<Z3> = LazyHyperoctahedralSpecies(QQ, 3)
            sage: (A + B)(X, Z3)
            Traceback (most recent call last):
            ...
            ValueError: unable to find a common parent for (A+B) and the substituted r-species (X, Z3)
        """
        fP = left.parent()
        # Find a good parent for the result
        cm = get_coercion_model()
        try:
            P = cm.common_parent(left.base_ring(), *[parent(g) for g in args])
        except TypeError:
            raise ValueError(f"unable to find a common parent for {left} "
                            f"and the substituted r-species {args}")
        if not isinstance(P, LazyHyperoctahedralSpecies):
            raise ValueError(f"the substituted species {args} must be "
                             f"lazy r-species with the same r and sorts")

        args = [P(g) for g in args]

        R = P._internal_poly_ring.base_ring()
        molecules = R._indices

        # f = 0
        if isinstance(left._coeff_stream, Stream_zero):
            super().__init__(P, Stream_zero())
            self._left = left
            self._args = args
            return

        # all substituted species are zero, so the composition is constant
        if all(isinstance(g._coeff_stream, Stream_zero) for g in args):
            c = left[0]
            if c:
                c = list(c)[0][1]
            else:
                c = P.base_ring().zero()
            super().__init__(P, P(c)._coeff_stream)
            self._left = left
            self._args = args
            return

        # the outer species is a constant polynomial
        if (isinstance(left._coeff_stream, Stream_exact)
                and not left._coeff_stream._constant
                and left._coeff_stream._degree == 1):
            c = left._coeff_stream[0]
            B = c.parent()
            if not (B is ZZ or B is QQ or B == fP.base_ring()):
                c = c.coefficients()[0]
            super().__init__(P, P(c)._coeff_stream)
            self._left = left
            self._args = args
            return

        for g in args:
            if g._coeff_stream._approximate_order == 0:
                if not g._coeff_stream.is_uninitialized() and g[0]:
                    raise ValueError("can only compose with a positive valuation series")
                g._coeff_stream._approximate_order = 1

        sorder = left._coeff_stream._approximate_order
        gv = min(g._coeff_stream._approximate_order for g in args)
        L = fP._internal_poly_ring.base_ring()

        def coeff(g, i):
            c = g._coeff_stream[i]
            if not isinstance(c, PolynomialSpecies.Element):
                return R(c)
            return c

        # args_flat and weights contain one list for each substituted species
        weight_exp = [lazy_list(lambda j, g=g: len(coeff(g, j+1)))
                      for g in args]

        def flat(g):
            # function needed to work around python's scoping rules
            return itertools.chain.from_iterable(coeff(g, j) for j in itertools.count())

        args_flat1 = [lazy_list(flat(g)) for g in args]

        def coefficient(n):
            if not n:
                if left[0]:
                    return R(list(left[0])[0][1])
                return R.zero()
            result = R.zero()
            for i in range(1, n // gv + 1):
                # skip i=0 because it produces a term only for n=0

                # compute homogeneous components
                lF = defaultdict(L)
                for M, c in left[i]:
                    lF[M.grade()] += L._from_dict({M: c})
                for mc, F in lF.items():
                    for degrees in weighted_vector_compositions(mc, n, weight_exp):
                        args_flat = [list(a[0:len(degrees[j])])
                                     for j, a in enumerate(args_flat1)]
                        multiplicities = [c for alpha, g_flat in zip(degrees, args_flat)
                                          for d, (_, c) in zip(alpha, g_flat) if d]
                        subs = [M for alpha, g_flat in zip(degrees, args_flat)
                                for d, (M, _) in zip(alpha, g_flat) if d]
                        non_zero_degrees = [[d for d in alpha if d] for alpha in degrees]
                        names = ["X%s" % i for i in range(len(subs))]
                        FX = F._compose_with_weighted_singletons(names,
                                                                 multiplicities,
                                                                 non_zero_degrees)
                        FG = [(molecules._type1_substitute_molecular(M, subs), c)
                              for M, c in FX]
                        result += R.sum_of_terms(FG)
            return result

        coeff_stream = Stream_function(coefficient, P._sparse, sorder * gv)
        super().__init__(P, coeff_stream)
        self._left = left
        self._args = args

    def generating_series(self):
        r"""
        Return the generating series of ``self``.

        The generating series of a type 1 substitution is the
        substitution of the generating series [Henderson2004, Section 4].

        EXAMPLES::

            sage: from sage.rings.lazy_species_hyperoctahedral import LazyHyperoctahedralSpecies
            sage: from sage.rings.lazy_species import LazyCombinatorialSpecies
            sage: from sage.groups.perm_gps.hyperoctahedral_group import _wreath_group
            sage: L.<X> = LazyHyperoctahedralSpecies(QQ, 2)
            sage: Xo = L(_wreath_group(2, 1))
            sage: E = LazyCombinatorialSpecies(QQ, "Z").Sets()
            sage: E(X + Xo).generating_series()
            1 + 3/2*X + 9/8*X^2 + 9/16*X^3 + 27/128*X^4 + 81/1280*X^5
             + 81/5120*X^6 + O(X^7)
            sage: exp(Xo.generating_series())[:7] == E(Xo).generating_series()[:7]
            True
        """
        return self._left.generating_series()(*[G.generating_series()
                                                for G in self._args])


class LazyHyperoctahedralSpecies(LazyCompletionGradedAlgebra):
    r"""
    The ring of lazy `r`-species.

    This is the completion of the graded algebra of polynomial
    `r`-species, where coefficients of degree `n` are finite linear
    combinations of molecular `r`-species of degree `n`, i.e., of
    `r`-species with `n` `C_r`-blocks.

    INPUT:

    - ``base_ring`` -- the base ring

    - ``r`` -- positive integer; the order of the cyclic group `C_r`

    - ``names`` -- names of the sorts

    - ``sparse`` -- boolean (default: ``True``); whether we use a sparse
      or a dense representation

    EXAMPLES::

        sage: from sage.rings.lazy_species_hyperoctahedral import LazyHyperoctahedralSpecies
        sage: LazyHyperoctahedralSpecies(QQ, 2)
        Lazy completion of Polynomial 2-species over Rational Field
        sage: LazyHyperoctahedralSpecies(QQ, 2, "X, Y")
        Lazy completion of Polynomial 2-species in X, Y over Rational Field

    TESTS::

        sage: LazyHyperoctahedralSpecies(QQ, 2, "X") is LazyHyperoctahedralSpecies(QQ, 2, "X")
        True
    """

    Element = LazyHyperoctahedralSpeciesElement

    @staticmethod
    def __classcall_private__(cls, base_ring, r, names="X", sparse=True):
        r"""
        Normalize input to ensure a unique representation.
        """
        from sage.structure.category_object import normalize_names
        names = normalize_names(-1, names)
        return super().__classcall__(cls, base_ring, ZZ(r), names, sparse)

    def _first_ngens(self, n):
        r"""
        Used by the preparser for ``F.<x> = ...``.

        We do not use the generic implementation of
        :class:`sage.combinat.CombinatorialFreeModule`, because we do
        not want to implement `gens`.

        Only the first ``n`` degree one species are returned::

            sage: from sage.rings.lazy_species_hyperoctahedral import LazyHyperoctahedralSpecies
            sage: L = LazyHyperoctahedralSpecies(QQ, 2, "X, Y")
            sage: L._first_ngens(2)
            (X, Y)
        """
        return tuple(self(g) for g in self._laurent_poly_ring._first_ngens(n))

    def __init__(self, base_ring, r, names, sparse):
        r"""
        Initialize the ring of lazy `r`-species.

        TESTS::

            sage: from sage.rings.lazy_species_hyperoctahedral import LazyHyperoctahedralSpecies
            sage: LazyHyperoctahedralSpecies(QQ, 2)._arity
            1
            sage: LazyHyperoctahedralSpecies(QQ, 2, "X, Y")._arity
            2
        """
        super().__init__(PolynomialHyperoctahedralSpecies(base_ring, r, names),
                         sparse=sparse)
        self._arity = len(names)
