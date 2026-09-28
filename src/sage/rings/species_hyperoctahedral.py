r"""
Polynomial `r`-species

This module is the analogue of :mod:`sage.rings.species` for polynomial
`r`-species in the sense of Henderson [Henderson2004]_, i.e. functors on
the category `\mathbf B_r` of finite sets with a free action of the
cyclic group `C_r` of order `r`.  The case `r = 2` is the
*hyperoctahedral* case.

The combinatorial model of the canonical permutation representation of
the wreath product `W(r,n) = C_r \wr S_n` lives in
:mod:`sage.groups.perm_gps.hyperoctahedral_group`; the cycle-index
algebra `\Lambda(r)` lives in :mod:`sage.rings.sf_hyperoctahedral`.

A *molecular* `r`-species of degree `n` is a transitive `W(r,n)`-set,
represented by a subgroup `H \leq W(r,n)` up to conjugacy *inside*
`W(r,n)`.  It is important that conjugacy is never taken in the larger
symmetric group `S_{rn}`.

EXAMPLES:

Atomic `r`-species are represented by subgroups of `W(r,n)` up to
`W(r,n)`-conjugacy.  For `r = 2` and `n = 1` there are two of them,
the trivial species `X` and the sign species `X°`::

    sage: from sage.rings.species_hyperoctahedral import (
    ....:     AtomicHyperoctahedralSpecies, MolecularHyperoctahedralSpecies)
    sage: from sage.groups.perm_gps.hyperoctahedral_group import _wreath_group
    sage: A = AtomicHyperoctahedralSpecies(2)
    sage: W = _wreath_group(2, 1)
    sage: A(W.subgroup([]))
    X
    sage: A(W)
    X°

For larger `r` there is one degree-one species for every divisor `d`
of `r`, displayed as ``X@d``, where `d` is the order of the stabilizer::

    sage: for r in [1, 2, 4, 6]:
    ....:     A = AtomicHyperoctahedralSpecies(r)
    ....:     print(sorted(A.graded_component(1), key=str))
    [X]
    [X, X°]
    [X, X@2, X°]
    [X, X@2, X@3, X°]

Molecular `r`-species are the products of atomic ones.  They enumerate
the conjugacy classes of subgroups of `W(r,n)`::

    sage: M = MolecularHyperoctahedralSpecies(2)
    sage: W = _wreath_group(2, 2)
    sage: M(W.subgroup([]))
    X^2
    sage: M(W.subgroup([(1, 2), (3, 4)]))
    X°^2
    sage: M(W.subgroup([(1, 2)]))
    X°*X

Conjugacy is taken inside `W(r,n)`: the two sign subgroups of `W(2,2)`
are conjugate, while they would not be conjugate in `S_4`::

    sage: M(W.subgroup([(1, 2)])) == M(W.subgroup([(3, 4)]))
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

from sage.arith.misc import divisors
from sage.categories.monoids import Monoids
from sage.categories.sets_with_grading import SetsWithGrading
from sage.groups.perm_gps.constructor import PermutationGroupElement
from sage.groups.perm_gps.hyperoctahedral_group import (
    _wreath_group,
    _hyperoctahedral_disjoint_direct_product_decomposition,
)
from sage.groups.perm_gps.permgroup import PermutationGroup, PermutationGroup_generic
from sage.libs.gap.libgap import libgap
from sage.misc.cachefunc import cached_function, cached_method
from sage.misc.fast_methods import WithEqualityById
from sage.misc.inherit_comparison import InheritComparisonClasscallMetaclass
from sage.monoids.indexed_free_monoid import (IndexedFreeAbelianMonoid,
                                              IndexedFreeAbelianMonoidElement)
from sage.rings.integer import Integer
from sage.rings.integer_ring import ZZ
from sage.sets.non_negative_integers import NonNegativeIntegers
from sage.sets.set import Set
from sage.structure.element import Element, parent
from sage.structure.parent import Parent
from sage.structure.unique_representation import (UniqueRepresentation,
                                                  WithPicklingByInitArgs)

GAP_FAIL = libgap.eval('fail')


@cached_function
def _wreath_subgroup_classes(r, n):
    r"""
    Return representatives of the conjugacy classes of subgroups of `W(r,n)`.

    INPUT:

    - ``r`` -- positive integer; the order of the cyclic group
    - ``n`` -- nonnegative integer

    EXAMPLES::

        sage: from sage.rings.species_hyperoctahedral import _wreath_subgroup_classes
        sage: len(_wreath_subgroup_classes(2, 2))
        8
        sage: all(G.degree() == 4 for G in _wreath_subgroup_classes(2, 2))
        True
    """
    return _wreath_group(r, n).conjugacy_classes_subgroups()


@cached_function
def _wreath_subgroup_classes_by_order(r, n):
    r"""
    Return the `W(r,n)`-subgroup classes grouped by order.

    EXAMPLES::

        sage: from sage.rings.species_hyperoctahedral import _wreath_subgroup_classes_by_order
        sage: d = _wreath_subgroup_classes_by_order(2, 2)
        sage: sorted(d)
        [1, 2, 4, 8]
        sage: [len(v) for k, v in sorted(d.items())]
        [1, 3, 3, 1]
    """
    result = {}
    for idx, rep in enumerate(_wreath_subgroup_classes(r, n)):
        result.setdefault(rep.order(), []).append((idx, rep))
    return result


@cached_function
def _wreath_subgroup_class_id_to_index(r, n):
    r"""
    Return the map from subgroup class representatives to their index.

    The keys are the (object) identities of the representatives returned
    by :func:`_wreath_subgroup_classes`; these objects are kept alive by
    the cache of that function.

    EXAMPLES::

        sage: from sage.rings.species_hyperoctahedral import (
        ....:     _wreath_subgroup_classes, _wreath_subgroup_class_id_to_index)
        sage: d = _wreath_subgroup_class_id_to_index(2, 2)
        sage: len(d)
        8
        sage: all(d[id(rep)] == idx
        ....:     for idx, rep in enumerate(_wreath_subgroup_classes(2, 2)))
        True
    """
    return {id(rep): idx
            for idx, rep in enumerate(_wreath_subgroup_classes(r, n))}


def _canonical_wreath_subgroup_index(G, r):
    r"""
    Return the index of the `W(r,n)`-conjugacy class of ``G``.

    The subgroup ``G`` must be a subgroup of `W(r,n)` acting on
    `\{1, \ldots, rn\}` with the standard consecutive block system.

    INPUT:

    - ``G`` -- a permutation group
    - ``r`` -- positive integer; the order of the cyclic group

    OUTPUT:

    A pair ``(index, representative)``, where ``representative`` is the
    (unique) representative of the `W(r,n)`-conjugacy class of ``G`` in
    :func:`_wreath_subgroup_classes`.

    EXAMPLES::

        sage: from sage.rings.species_hyperoctahedral import _canonical_wreath_subgroup_index
        sage: from sage.groups.perm_gps.hyperoctahedral_group import _wreath_group
        sage: W = _wreath_group(2, 2)
        sage: G = W.subgroup([(1, 2)])
        sage: idx, rep = _canonical_wreath_subgroup_index(G, 2)
        sage: rep.is_subgroup(W)
        True
        sage: _canonical_wreath_subgroup_index(rep, 2)[0] == idx
        True
    """
    n = G.degree() // r
    W = _wreath_group(r, n)
    W_gap = W.gap()
    G_gap = G.gap()
    classes = _wreath_subgroup_classes(r, n)
    by_id = _wreath_subgroup_class_id_to_index(r, n)
    if id(G) in by_id:
        idx = by_id[id(G)]
        return idx, classes[idx]
    for idx, rep in _wreath_subgroup_classes_by_order(r, n).get(G.order(), []):
        if libgap.RepresentativeAction(W_gap, G_gap, rep.gap()) != GAP_FAIL:
            return idx, rep
    raise ValueError(f"{G} is not conjugate to a subgroup of {W}")


def _standardize_component(H, comp, r):
    r"""
    Return the component ``comp`` of ``H`` as a subgroup of `W(r,k)`.

    The set ``comp`` must be a union of complete consecutive `C_r`-blocks
    of the standard domain of `H`.  Its points are relabelled to
    `\{1, \ldots, rk\}` in order, which is a block-respecting bijection.

    INPUT:

    - ``H`` -- a permutation group on a union of complete `C_r`-blocks
    - ``comp`` -- an iterable; the subset of the domain to restrict to
    - ``r`` -- positive integer; the order of the cyclic group

    OUTPUT:

    A permutation group on `\{1, \ldots, rk\}`.

    EXAMPLES::

        sage: from sage.rings.species_hyperoctahedral import _standardize_component
        sage: from sage.groups.perm_gps.hyperoctahedral_group import _wreath_group
        sage: H = _wreath_group(2, 3).subgroup([(3, 4)])
        sage: K = _standardize_component(H, {3, 4}, 2)
        sage: K.degree()
        2
        sage: K == _wreath_group(2, 1)
        True
    """
    comp = sorted(comp)

    gens = []
    for gen in H.gens():
        cycles = [cyc for cyc in gen.cycle_tuples() if cyc[0] in comp]
        if cycles:
            gens.append(PermutationGroupElement(cycles))
    Hc = PermutationGroup(gens, domain=comp)

    relabel = {p: i + 1 for i, p in enumerate(comp)}
    new_gens = []
    for g in Hc.gens():
        perm = list(range(1, len(comp) + 1))
        for p in comp:
            perm[relabel[p] - 1] = relabel[g(p)]
        new_gens.append(PermutationGroupElement(perm))
    return PermutationGroup(new_gens, domain=range(1, len(comp) + 1))


def _check_standard_domain(G, r):
    r"""
    Check that ``G`` acts on the standard domain `\{1, \ldots, rn\}`.

    INPUT:

    - ``G`` -- a permutation group
    - ``r`` -- positive integer; the order of the cyclic group

    EXAMPLES::

        sage: from sage.rings.species_hyperoctahedral import _check_standard_domain
        sage: from sage.groups.perm_gps.hyperoctahedral_group import _wreath_group
        sage: _check_standard_domain(_wreath_group(2, 2), 2)
        sage: _check_standard_domain(_wreath_group(2, 2).subgroup([(1, 2)]), 2)

    A group on a non-standard domain is rejected::

        sage: from sage.groups.perm_gps.permgroup import PermutationGroup
        sage: G = PermutationGroup([(2, 3)], domain=[2, 3, 4, 5])
        sage: _check_standard_domain(G, 2)
        Traceback (most recent call last):
        ...
        ValueError: Permutation Group with generators [(2,3)] must act on the standard domain {1, ..., 4}
    """
    degree = G.degree()
    if degree % r:
        raise ValueError(f"the degree {degree} is not divisible by r = {r}")
    W = _wreath_group(r, degree // r)
    if set(G.domain()) != set(W.domain()):
        raise ValueError(f"{G} must act on the standard domain {{1, ..., {degree}}}")


class AtomicHyperoctahedralSpecies(UniqueRepresentation, Parent):
    r"""
    The set of atomic `r`-species.

    An atomic `r`-species of degree `n` is a directly indecomposable
    transitive `W(r,n)`-set, represented up to conjugacy inside
    `W(r,n)`.

    INPUT:

    - ``r`` -- positive integer; the order of the cyclic group

    EXAMPLES::

        sage: from sage.rings.species_hyperoctahedral import AtomicHyperoctahedralSpecies
        sage: A = AtomicHyperoctahedralSpecies(2)
        sage: A
        Atomic 2-species
        sage: A.grading_set()
        Non negative integers

    TESTS::

        sage: from sage.rings.species_hyperoctahedral import AtomicHyperoctahedralSpecies
        sage: AtomicHyperoctahedralSpecies(2) is AtomicHyperoctahedralSpecies(ZZ(2))
        True
        sage: AtomicHyperoctahedralSpecies(0)
        Traceback (most recent call last):
        ...
        ValueError: r must be a positive integer
    """
    @staticmethod
    def __classcall__(cls, r):
        r"""
        Normalize the arguments for unique representation.

        TESTS::

            sage: from sage.rings.species_hyperoctahedral import AtomicHyperoctahedralSpecies
            sage: AtomicHyperoctahedralSpecies(2) is AtomicHyperoctahedralSpecies(2)
            True
        """
        r = ZZ(r)
        if r < 1:
            raise ValueError("r must be a positive integer")
        return super().__classcall__(cls, r)

    def __init__(self, r):
        r"""
        Initialize the class of atomic `r`-species.

        EXAMPLES::

            sage: from sage.rings.species_hyperoctahedral import AtomicHyperoctahedralSpecies
            sage: A = AtomicHyperoctahedralSpecies(2)
            sage: A._r
            2
        """
        category = SetsWithGrading().Infinite()
        Parent.__init__(self, category=category)
        self._r = ZZ(r)
        self._cache = dict()
        # the degrees whose standard species have already been renamed
        self._renamed = set()

    def _repr_(self):
        r"""
        Return a string representation of ``self``.

        EXAMPLES::

            sage: from sage.rings.species_hyperoctahedral import AtomicHyperoctahedralSpecies
            sage: AtomicHyperoctahedralSpecies(3)
            Atomic 3-species
        """
        return f"Atomic {self._r}-species"

    def _rename(self, n):
        r"""
        Give the standard atomic `r`-species of degree ``n`` their names.

        For `n = 1` the atomic `r`-species correspond to the divisors
        `d \mid r`: the stabilizer of the species `C_r / C_d` is the
        cyclic group `C_d` of order `d`.  Following the convention of
        [Henderson2004]_, we display it as ``X@d``, with the extremal
        cases ``X = X@1`` and ``X° = X@r``, and with ``X`` for the
        unique species when `r = 1`.

        EXAMPLES::

            sage: from sage.rings.species_hyperoctahedral import AtomicHyperoctahedralSpecies
            sage: from sage.groups.perm_gps.hyperoctahedral_group import _wreath_group
            sage: A = AtomicHyperoctahedralSpecies(2)
            sage: A(_wreath_group(2, 1))
            X°
        """
        if n != 1:
            return
        W = _wreath_group(self._r, 1)
        cycle = W.gens()[0]
        for d in divisors(self._r):
            G = W.subgroup([cycle ** (self._r // d)])
            if d == 1:
                name = "X"
            elif d == self._r:
                name = "X°"
            else:
                name = f"X@{d}"
            self(G, check=False).rename(name)

    def _element_constructor_(self, G, check=True):
        r"""
        Construct the atomic `r`-species given by the subgroup ``G``.

        INPUT:

        - ``G`` -- a permutation group which is a subgroup of `W(r,n)`
          on the standard domain, or an element of ``self``
        - ``check`` -- boolean (default: ``True``); whether to check
          that ``G`` is directly indecomposable

        EXAMPLES::

            sage: from sage.rings.species_hyperoctahedral import AtomicHyperoctahedralSpecies
            sage: from sage.groups.perm_gps.hyperoctahedral_group import _wreath_group
            sage: A = AtomicHyperoctahedralSpecies(2)
            sage: G = _wreath_group(2, 2)
            sage: A(G.subgroup([[(1, 3), (2, 4)]]))
            {((1,3)(2,4),)}
            sage: A(G.subgroup([(1, 2), (3, 4)]))
            Traceback (most recent call last):
            ...
            ValueError: ((1,2), (3,4)) is not directly indecomposable
        """
        if parent(G) is self:
            return G
        if not isinstance(G, PermutationGroup_generic):
            raise ValueError(f"{G} must be a permutation group")
        _check_standard_domain(G, self._r)
        if check and len(_hyperoctahedral_disjoint_direct_product_decomposition(G, self._r)) != 1:
            raise ValueError(f"{G.gens()} is not directly indecomposable")
        return self.element_class(self, G)

    def grading_set(self):
        r"""
        Return the grading set of ``self``.

        EXAMPLES::

            sage: from sage.rings.species_hyperoctahedral import AtomicHyperoctahedralSpecies
            sage: AtomicHyperoctahedralSpecies(2).grading_set()
            Non negative integers
        """
        return NonNegativeIntegers()

    def subset(self, size):
        r"""
        Return the set of atomic `r`-species of degree ``size``.

        EXAMPLES::

            sage: from sage.rings.species_hyperoctahedral import AtomicHyperoctahedralSpecies
            sage: A = AtomicHyperoctahedralSpecies(2)
            sage: sorted(a.degree() for a in A.subset(2))
            [2, 2, 2, 2, 2]
        """
        return self.graded_component(size)

    def graded_component(self, n):
        r"""
        Return the set of atomic `r`-species of degree ``n``.

        EXAMPLES::

            sage: from sage.rings.species_hyperoctahedral import AtomicHyperoctahedralSpecies
            sage: A = AtomicHyperoctahedralSpecies(2)
            sage: len(A.graded_component(2))
            5
        """
        return Set([self(rep) for rep in _wreath_subgroup_classes(self._r, n)
                    if len(_hyperoctahedral_disjoint_direct_product_decomposition(rep, self._r)) == 1])

    def __contains__(self, x):
        r"""
        Return whether ``x`` is an atomic `r`-species, or a subgroup
        defining one.

        EXAMPLES::

            sage: from sage.rings.species_hyperoctahedral import AtomicHyperoctahedralSpecies
            sage: from sage.groups.perm_gps.hyperoctahedral_group import _wreath_group
            sage: A = AtomicHyperoctahedralSpecies(1)
            sage: _wreath_group(1, 1).subgroup([]) in A
            True
        """
        if parent(x) is self:
            return True
        if not isinstance(x, PermutationGroup_generic):
            return False
        degree = x.degree()
        if degree % self._r:
            return False
        W = _wreath_group(self._r, degree // self._r)
        if set(x.domain()) != set(W.domain()):
            return False
        if libgap.IsSubgroup(W.gap(), x.gap()) != True:
            return False
        return len(_hyperoctahedral_disjoint_direct_product_decomposition(x, self._r)) == 1

    class Element(WithEqualityById,
                  Element,
                  WithPicklingByInitArgs,
                  metaclass=InheritComparisonClasscallMetaclass):
        r"""
        An atomic `r`-species.
        """
        @staticmethod
        def __classcall__(cls, parent, G):
            r"""
            Normalize the input for unique representation.

            TESTS::

                sage: from sage.rings.species_hyperoctahedral import AtomicHyperoctahedralSpecies
                sage: from sage.groups.perm_gps.hyperoctahedral_group import _wreath_group
                sage: A = AtomicHyperoctahedralSpecies(2)
                sage: G = _wreath_group(2, 2)
                sage: A(G.subgroup([[(1, 3), (2, 4)]])) is A(G.subgroup([[(1, 3), (2, 4)]]))
                True
            """
            r = parent._r
            idx, rep = _canonical_wreath_subgroup_index(G, r)
            key = (G.degree() // r, idx)
            if key in parent._cache:
                return parent._cache[key]
            elm = WithPicklingByInitArgs.__classcall__(cls, parent, rep)
            parent._cache[key] = elm
            return elm

        def __init__(self, parent, G):
            r"""
            Initialize an atomic `r`-species.

            EXAMPLES::

                sage: from sage.rings.species_hyperoctahedral import AtomicHyperoctahedralSpecies
                sage: from sage.groups.perm_gps.hyperoctahedral_group import _wreath_group
                sage: A = AtomicHyperoctahedralSpecies(2)
                sage: a = A(_wreath_group(2, 2).subgroup([[(1, 3), (2, 4)]]))
                sage: TestSuite(a).run()
                sage: loads(dumps(a)) is a
                True
            """
            Element.__init__(self, parent)
            self._dis = G
            self._n = G.degree() // parent._r

        def _repr_(self):
            r"""
            Return a string representation of ``self``.

            The first time a degree is displayed, the standard names
            for that degree are installed by :meth:`_rename`.

            EXAMPLES::

                sage: from sage.rings.species_hyperoctahedral import AtomicHyperoctahedralSpecies
                sage: from sage.groups.perm_gps.hyperoctahedral_group import _wreath_group
                sage: A = AtomicHyperoctahedralSpecies(2)
                sage: A(_wreath_group(2, 2).subgroup([[(1, 3), (2, 4)]]))
                {((1,3)(2,4),)}
            """
            P = self.parent()
            if self._n not in P._renamed:
                P._renamed.add(self._n)
                P._rename(self._n)
                return repr(self)
            return "{" + f"{self._dis.gens()}" + "}"

        def permutation_group(self):
            r"""
            Return the permutation group representing ``self``.

            EXAMPLES::

                sage: from sage.rings.species_hyperoctahedral import AtomicHyperoctahedralSpecies
                sage: from sage.groups.perm_gps.hyperoctahedral_group import _wreath_group
                sage: A = AtomicHyperoctahedralSpecies(2)
                sage: A(_wreath_group(2, 1)).permutation_group()
                Subgroup generated by [(1,2)] of (Permutation Group with generators [(1,2)])
            """
            return self._dis

        def degree(self):
            r"""
            Return the degree of ``self``.

            EXAMPLES::

                sage: from sage.rings.species_hyperoctahedral import AtomicHyperoctahedralSpecies
                sage: from sage.groups.perm_gps.hyperoctahedral_group import _wreath_group
                sage: A = AtomicHyperoctahedralSpecies(2)
                sage: A(_wreath_group(2, 2).subgroup([[(1, 3), (2, 4)]])).degree()
                2
            """
            return self._n

        def grade(self):
            r"""
            Return the grade of ``self``.

            EXAMPLES::

                sage: from sage.rings.species_hyperoctahedral import AtomicHyperoctahedralSpecies
                sage: from sage.groups.perm_gps.hyperoctahedral_group import _wreath_group
                sage: A = AtomicHyperoctahedralSpecies(2)
                sage: A(_wreath_group(2, 2).subgroup([[(1, 3), (2, 4)]])).grade()
                2
            """
            return self._n

        def is_atomic(self):
            r"""
            Return whether ``self`` is atomic, which is always the case.

            EXAMPLES::

                sage: from sage.rings.species_hyperoctahedral import AtomicHyperoctahedralSpecies
                sage: from sage.groups.perm_gps.hyperoctahedral_group import _wreath_group
                sage: A = AtomicHyperoctahedralSpecies(2)
                sage: A(_wreath_group(2, 2).subgroup([[(1, 3), (2, 4)]])).is_atomic()
                True
            """
            return True

    Element = Element


class MolecularHyperoctahedralSpecies(IndexedFreeAbelianMonoid):
    r"""
    The monoid of molecular `r`-species.

    This is the commutative free abelian monoid generated by the atomic
    `r`-species, exactly as :class:`~sage.rings.species.MolecularSpecies`
    is generated by :class:`~sage.rings.species.AtomicSpecies`.

    INPUT:

    - ``r`` -- positive integer; the order of the cyclic group

    EXAMPLES::

        sage: from sage.rings.species_hyperoctahedral import MolecularHyperoctahedralSpecies
        sage: M = MolecularHyperoctahedralSpecies(2)
        sage: M
        Molecular 2-species

    TESTS::

        sage: from sage.rings.species_hyperoctahedral import MolecularHyperoctahedralSpecies
        sage: MolecularHyperoctahedralSpecies(2) is MolecularHyperoctahedralSpecies(ZZ(2))
        True
    """
    @staticmethod
    def __classcall__(cls, r):
        r"""
        Normalize the arguments for unique representation.

        TESTS::

            sage: from sage.rings.species_hyperoctahedral import MolecularHyperoctahedralSpecies
            sage: MolecularHyperoctahedralSpecies(2) is MolecularHyperoctahedralSpecies(2)
            True
        """
        r = ZZ(r)
        if r < 1:
            raise ValueError("r must be a positive integer")
        return UniqueRepresentation.__classcall__(cls, r)

    def __init__(self, r):
        r"""
        Initialize the monoid of molecular `r`-species.

        EXAMPLES::

            sage: from sage.rings.species_hyperoctahedral import MolecularHyperoctahedralSpecies
            sage: M = MolecularHyperoctahedralSpecies(2)
            sage: M._r
            2
        """
        indices = AtomicHyperoctahedralSpecies(r)
        category = Monoids().Commutative() & SetsWithGrading().Infinite()
        IndexedFreeAbelianMonoid.__init__(self, indices, prefix='',
                                          bracket=False, category=category)
        self._r = ZZ(r)

    def _repr_(self):
        r"""
        Return a string representation of ``self``.

        EXAMPLES::

            sage: from sage.rings.species_hyperoctahedral import MolecularHyperoctahedralSpecies
            sage: MolecularHyperoctahedralSpecies(3)
            Molecular 3-species
        """
        return f"Molecular {self._r}-species"

    def _element_constructor_(self, G, check=True):
        r"""
        Construct the molecular `r`-species given by the subgroup ``G``.

        INPUT:

        - ``G`` -- a permutation group which is a subgroup of `W(r,n)`
          on the standard domain, or an element of ``self``, or a
          dictionary from atoms to exponents
        - ``check`` -- boolean (default: ``True``); whether to check
          the dictionary input

        EXAMPLES:

        The trivial subgroup of `W(2,2)` is the square of the trivial
        degree-one `r`-species::

            sage: from sage.rings.species_hyperoctahedral import MolecularHyperoctahedralSpecies
            sage: from sage.groups.perm_gps.hyperoctahedral_group import _wreath_group
            sage: M = MolecularHyperoctahedralSpecies(2)
            sage: M(_wreath_group(2, 2).subgroup([]))
            X^2

        The subgroup generated by the two sign changes is the product of
        two copies of the sign species::

            sage: M(_wreath_group(2, 2).subgroup([(1, 2), (3, 4)]))
            X°^2

        The subgroup generated by the block interchange is atomic::

            sage: M(_wreath_group(2, 2).subgroup([[(1, 3), (2, 4)]]))
            {((1,3)(2,4),)}

        A single sign change acts on one block and fixes the other::

            sage: M(_wreath_group(2, 2).subgroup([(1, 2)]))
            X°*X
        """
        if parent(G) is self:
            raise ValueError("cannot reassign data to a molecular species")
        if isinstance(G, dict):
            if check:
                if not all(A.parent() is self._indices for A in G):
                    raise ValueError(f"all keys of the dict {G} must be {self._indices}")
                if not all(isinstance(e, Integer) for e in G.values()):
                    raise ValueError(f"all values of the dict {G} must be Integers")
            return self.element_class(self, G)
        if not isinstance(G, PermutationGroup_generic):
            raise ValueError(f"{G} must be a permutation group")
        _check_standard_domain(G, self._r)

        decomposition = _hyperoctahedral_disjoint_direct_product_decomposition(G, self._r)
        result = self.one()
        for comp in decomposition:
            H = _standardize_component(G, comp, self._r)
            result *= self.gen(self._indices(H))
        return result

    def grading_set(self):
        r"""
        Return the grading set of ``self``.

        EXAMPLES::

            sage: from sage.rings.species_hyperoctahedral import MolecularHyperoctahedralSpecies
            sage: MolecularHyperoctahedralSpecies(2).grading_set()
            Non negative integers
        """
        return NonNegativeIntegers()

    def subset(self, size):
        r"""
        Return the set of molecular `r`-species of degree ``size``.

        EXAMPLES::

            sage: from sage.rings.species_hyperoctahedral import MolecularHyperoctahedralSpecies
            sage: M = MolecularHyperoctahedralSpecies(2)
            sage: len(M.subset(2))
            8
        """
        return self.graded_component(size)

    def graded_component(self, n):
        r"""
        Return the set of molecular `r`-species of degree ``n``.

        EXAMPLES::

            sage: from sage.rings.species_hyperoctahedral import MolecularHyperoctahedralSpecies
            sage: M = MolecularHyperoctahedralSpecies(2)
            sage: len(M.graded_component(2))
            8

        The number of molecular `r`-species of degree `n` is the
        number of conjugacy classes of subgroups of `W(r,n)`::

            sage: M = MolecularHyperoctahedralSpecies(2)
            sage: [len(M.graded_component(n)) for n in range(1, 5)]
            [2, 8, 33, 193]

        Since the molecular species form the free commutative monoid on
        the atomic species, the atomic numbers can be recovered from the
        molecular numbers by the Euler transform::

            sage: from sage.arith.misc import divisors, moebius
            sage: from sage.rings.power_series_ring import PowerSeriesRing
            sage: from sage.rings.species_hyperoctahedral import AtomicHyperoctahedralSpecies
            sage: M = MolecularHyperoctahedralSpecies(2)
            sage: m = [ZZ(1)] + [len(M.graded_component(n)) for n in range(1, 5)]
            sage: R.<x> = PowerSeriesRing(QQ, default_prec=5)
            sage: f = R(m)
            sage: b = [ZZ(0)] + [n * f.log()[n] for n in range(1, 5)]
            sage: a = [sum(moebius(d) * b[n // d] for d in divisors(n)) / n
            ....:      for n in range(1, 5)]
            sage: a
            [2, 5, 19, 120]
            sage: A = AtomicHyperoctahedralSpecies(2)
            sage: [len(A.graded_component(n)) for n in range(1, 5)] == a
            True
        """
        return Set([self(rep) for rep in _wreath_subgroup_classes(self._r, n)])

    class Element(IndexedFreeAbelianMonoidElement):
        r"""
        A molecular `r`-species.
        """
        @cached_method
        def grade(self):
            r"""
            Return the grade of ``self``.

            EXAMPLES::

                sage: from sage.rings.species_hyperoctahedral import MolecularHyperoctahedralSpecies
                sage: from sage.groups.perm_gps.hyperoctahedral_group import _wreath_group
                sage: M = MolecularHyperoctahedralSpecies(2)
                sage: M(_wreath_group(2, 2).subgroup([(1, 2), (3, 4)])).grade()
                2
            """
            return self.parent().grading_set()(
                sum(n * a.degree() for a, n in self._monomial.items()))

        def degree(self):
            r"""
            Return the degree of ``self``.

            EXAMPLES::

                sage: from sage.rings.species_hyperoctahedral import MolecularHyperoctahedralSpecies
                sage: from sage.groups.perm_gps.hyperoctahedral_group import _wreath_group
                sage: M = MolecularHyperoctahedralSpecies(2)
                sage: M(_wreath_group(2, 2).subgroup([(1, 2), (3, 4)])).degree()
                2
            """
            return Integer(sum(n * a.degree() for a, n in self._monomial.items()))

        def is_molecular(self):
            r"""
            Return whether ``self`` is molecular, which is always the case.

            EXAMPLES::

                sage: from sage.rings.species_hyperoctahedral import MolecularHyperoctahedralSpecies
                sage: from sage.groups.perm_gps.hyperoctahedral_group import _wreath_group
                sage: M = MolecularHyperoctahedralSpecies(2)
                sage: M(_wreath_group(2, 2).subgroup([[(1, 3), (2, 4)]])).is_molecular()
                True
            """
            return True

        def is_atomic(self):
            r"""
            Return whether ``self`` consists of a single atomic species.

            EXAMPLES::

                sage: from sage.rings.species_hyperoctahedral import MolecularHyperoctahedralSpecies
                sage: from sage.groups.perm_gps.hyperoctahedral_group import _wreath_group
                sage: M = MolecularHyperoctahedralSpecies(2)
                sage: M(_wreath_group(2, 2).subgroup([[(1, 3), (2, 4)]])).is_atomic()
                True
                sage: M(_wreath_group(2, 2).subgroup([(1, 2), (3, 4)])).is_atomic()
                False
            """
            return len(self._monomial) == 1 and next(iter(self._monomial.values())) == 1
