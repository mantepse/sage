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

Multisort `r`-species of grade `(n_1,\ldots,n_k)` are transitive sets
for the wreath Young subgroup `W(r;n_1,\ldots,n_k)`.  For example, the
diagonal `C_2` acting simultaneously on a block of each sort is an
atomic species of grade `(1,1)`, because it is directly
indecomposable::

    sage: from sage.groups.perm_gps.hyperoctahedral_group import _wreath_young_subgroup
    sage: A = AtomicHyperoctahedralSpecies(2, "X, Y")
    sage: W = _wreath_young_subgroup(2, [1, 1])
    sage: a = A(W.subgroup([[(1, 2), (3, 4)]]), {0: [1, 2], 1: [3, 4]})
    sage: a.grade()
    [1, 1]
    sage: a.is_atomic()
    True

By contrast, the full product `C_2 \times C_2` of the sign changes on
each sort decomposes into two molecular factors::

    sage: M = MolecularHyperoctahedralSpecies(2, "X, Y")
    sage: M(W.subgroup([(1, 2), (3, 4)]), {0: [1, 2], 1: [3, 4]})
    X°*Y°
    sage: M(W.subgroup([]), {0: [1, 2], 1: [3, 4]})
    X*Y

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

from itertools import chain

from sage.arith.misc import divisors
from sage.categories.graded_algebras_with_basis import GradedAlgebrasWithBasis
from sage.categories.monoids import Monoids
from sage.categories.sets_with_grading import SetsWithGrading
from sage.combinat.free_module import CombinatorialFreeModule
from sage.combinat.integer_vector import IntegerVectors
from sage.groups.perm_gps.constructor import PermutationGroupElement
from sage.groups.perm_gps.hyperoctahedral_group import (
    _check_grade,
    _wreath_blocks,
    _wreath_group,
    _wreath_young_subgroup,
    _wreath_young_subgroup_on_domain,
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
from sage.sets.set import Set
from sage.structure.category_object import normalize_names
from sage.structure.element import Element, parent
from sage.structure.parent import Parent
from sage.structure.unique_representation import (UniqueRepresentation,
                                                  WithPicklingByInitArgs)

GAP_FAIL = libgap.eval('fail')


@cached_function
def _wreath_young_subgroup_classes(r, grade):
    r"""
    Return representatives of the conjugacy classes of subgroups of
    `W(r;` ``grade`` `)`.

    INPUT:

    - ``r`` -- positive integer; the order of the cyclic group
    - ``grade`` -- a tuple of nonnegative integers, the number of
      `C_r`-blocks in each sort

    EXAMPLES::

        sage: from sage.rings.species_hyperoctahedral import _wreath_young_subgroup_classes
        sage: len(_wreath_young_subgroup_classes(2, (2,)))
        8
        sage: all(G.degree() == 4 for G in _wreath_young_subgroup_classes(2, (2,)))
        True
        sage: len(_wreath_young_subgroup_classes(2, (1, 1)))
        5
    """
    grade = _check_grade(grade)
    return _wreath_young_subgroup(r, grade).conjugacy_classes_subgroups()


@cached_function
def _wreath_young_subgroup_classes_by_order(r, grade):
    r"""
    Return the `W(r;` ``grade`` `)`-subgroup classes grouped by order.

    EXAMPLES::

        sage: from sage.rings.species_hyperoctahedral import _wreath_young_subgroup_classes_by_order
        sage: d = _wreath_young_subgroup_classes_by_order(2, (2,))
        sage: sorted(d)
        [1, 2, 4, 8]
        sage: [len(v) for k, v in sorted(d.items())]
        [1, 3, 3, 1]
    """
    result = {}
    for idx, rep in enumerate(_wreath_young_subgroup_classes(r, grade)):
        result.setdefault(rep.order(), []).append((idx, rep))
    return result


@cached_function
def _wreath_young_subgroup_class_id_to_index(r, grade):
    r"""
    Return the map from subgroup class representatives to their index.

    The keys are the (object) identities of the representatives returned
    by :func:`_wreath_young_subgroup_classes`; these objects are kept
    alive by the cache of that function.

    EXAMPLES::

        sage: from sage.rings.species_hyperoctahedral import (
        ....:     _wreath_young_subgroup_classes, _wreath_young_subgroup_class_id_to_index)
        sage: d = _wreath_young_subgroup_class_id_to_index(2, (2,))
        sage: len(d)
        8
        sage: all(d[id(rep)] == idx
        ....:     for idx, rep in enumerate(_wreath_young_subgroup_classes(2, (2,))))
        True
    """
    return {id(rep): idx
            for idx, rep in enumerate(_wreath_young_subgroup_classes(r, grade))}


def _canonical_wreath_subgroup_index(G, r, grade=None):
    r"""
    Return the index of the `W(r;` ``grade`` `)`-conjugacy class of ``G``.

    The subgroup ``G`` must act on `\{1, \ldots, rn\}` with the standard
    consecutive block system, where `n` is the total number of
    `C_r`-blocks.  If ``grade`` is not provided, it is assumed that all
    blocks belong to a single sort.

    INPUT:

    - ``G`` -- a permutation group
    - ``r`` -- positive integer; the order of the cyclic group
    - ``grade`` -- a tuple of nonnegative integers or ``None``
      (default); the number of `C_r`-blocks in each sort

    OUTPUT:

    A pair ``(index, representative)``, where ``representative`` is the
    (unique) representative of the `W(r;` ``grade`` `)`-conjugacy class
    of ``G`` in :func:`_wreath_young_subgroup_classes`.

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

    For several sorts the ambient group is the wreath Young subgroup::

        sage: from sage.groups.perm_gps.hyperoctahedral_group import _wreath_young_subgroup
        sage: W = _wreath_young_subgroup(2, [1, 1])
        sage: G = W.subgroup([(1, 2)])
        sage: idx, rep = _canonical_wreath_subgroup_index(G, 2, (1, 1))
        sage: rep.is_subgroup(W)
        True
        sage: rep.gens()
        ((1,2),)

    A group which is not a subgroup of the ambient wreath product is
    rejected::

        sage: G = PermutationGroup([(1, 3)])
        sage: _canonical_wreath_subgroup_index(G, 2, (2,))
        Traceback (most recent call last):
        ...
        ValueError: Permutation Group with generators [(1,3)] is not conjugate to a subgroup of Permutation Group with generators [(3,4), (1,2), (1,3)(2,4)]
    """
    if grade is None:
        grade = (G.degree() // r,)
    else:
        grade = _check_grade(grade)
    W = _wreath_young_subgroup(r, grade)
    W_gap = W.gap()
    G_gap = G.gap()
    classes = _wreath_young_subgroup_classes(r, grade)
    by_id = _wreath_young_subgroup_class_id_to_index(r, grade)
    if id(G) in by_id:
        idx = by_id[id(G)]
        return idx, classes[idx]
    for idx, rep in _wreath_young_subgroup_classes_by_order(r, grade).get(G.order(), []):
        if libgap.RepresentativeAction(W_gap, G_gap, rep.gap()) != GAP_FAIL:
            return idx, rep
    raise ValueError(f"{G} is not conjugate to a subgroup of {W}")


def _canonical_dompart(r, grade):
    r"""
    Return the canonical partition of the standard domain into sorts.

    The blocks of sort `s` are the consecutive blocks with numbers
    `n_1+\cdots+n_s+1, \ldots, n_1+\cdots+n_{s+1}`.

    INPUT:

    - ``r`` -- positive integer; the order of the cyclic group
    - ``grade`` -- a tuple of nonnegative integers, the number of
      `C_r`-blocks in each sort

    OUTPUT:

    A tuple of frozensets, the `s`-th entry consisting of the points
    of sort `s`.

    EXAMPLES::

        sage: from sage.rings.species_hyperoctahedral import _canonical_dompart
        sage: _canonical_dompart(2, (2, 1))
        (frozenset({1, 2, 3, 4}), frozenset({5, 6}))
        sage: _canonical_dompart(2, (0, 1))
        (frozenset(), frozenset({1, 2}))
    """
    return tuple(frozenset(range(r * sum(grade[:s]) + 1,
                                    r * sum(grade[:s + 1]) + 1))
                 for s in range(len(grade)))


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


def _wreath_stabilizers(X, a, side, pi, r, arity=1, check=True):
    r"""
    Return the stabilizers of an action of `W(r;n_1,\ldots,n_k)` on the
    set ``X``, together with the assignment of sorts.

    This is the `r`-species analogue of calling
    :func:`~sage.rings.species._stabilizer_subgroups` with a symmetric
    group: the acting group is the wreath Young subgroup
    `W(r;n_1,\ldots,n_k)` built on the domain specified by ``pi``, and
    the returned stabilizers are relabelled so that they act on the
    standard domain `\{1, \ldots, rn\}` with the standard block system.

    INPUT:

    - ``X`` -- the set of structures being acted on
    - ``a`` -- the action, cf.
      :func:`~sage.rings.species._stabilizer_subgroups`
    - ``side`` -- ``'left'`` or ``'right'``
    - ``pi`` -- a dictionary (or iterable) mapping sorts to domains; the
      domains must be unions of complete `C_r`-blocks
    - ``r`` -- positive integer; the order of the cyclic group
    - ``arity`` -- the number of sorts
    - ``check`` -- boolean (default: ``True``); whether to check that
      ``a`` is a group action

    OUTPUT:

    A list of pairs ``(H, pi_H)``, one for each orbit of the action,
    where ``H`` is a subgroup of `W(r,n)` on the standard domain and
    ``pi_H`` assigns the points of its domain to sorts.

    EXAMPLES::

        sage: from sage.rings.species_hyperoctahedral import _wreath_stabilizers
        sage: from sage.groups.perm_gps.hyperoctahedral_group import _wreath_group
        sage: W = _wreath_group(2, 2)
        sage: X = list(W.domain())
        sage: a = lambda g, x: g(x)
        sage: H = _wreath_stabilizers(X, a, 'left', {0: list(W.domain())}, 2)
        sage: len(H)
        1
        sage: H[0][0].order()
        2
        sage: H[0][1]
        {0: [1, 2, 3, 4]}

    For several sorts the acting group is the wreath Young subgroup,
    which has two orbits on its own domain::

        sage: H = _wreath_stabilizers(X, a, 'left', {0: [1, 2], 1: [3, 4]}, 2, arity=2)
        sage: len(H)
        2
        sage: H[0][0].order()
        2
        sage: H[0][1]
        {0: [1, 2], 1: [3, 4]}
    """
    from sage.rings.species import _stabilizer_subgroups
    if pi is None:
        raise ValueError("the assignment of sorts to the domain elements must be provided")
    if not isinstance(pi, dict):
        pi = dict(enumerate(pi))
    dompart = [sorted(pi.get(s, [])) for s in range(arity)]
    domain = list(chain.from_iterable(dompart))
    W = _wreath_young_subgroup_on_domain(dompart, r)
    stabilizers = _stabilizer_subgroups(W, X, a, side=side, check=check)

    relabel = {p: i + 1 for i, p in enumerate(sorted(domain))}
    return [(_standardize_component(H, H.domain(), r),
             {s: [relabel[p] for p in b] for s, b in enumerate(dompart)})
            for H in stabilizers]


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

    An atomic `r`-species of grade `(n_1,\ldots,n_k)` is a directly
    indecomposable transitive `W(r;n_1,\ldots,n_k)`-set, represented up
    to conjugacy inside `W(r;n_1,\ldots,n_k)`.

    INPUT:

    - ``r`` -- positive integer; the order of the cyclic group
    - ``names`` -- an iterable of strings for the sorts of the species
      (default: ``"X"``)

    EXAMPLES::

        sage: from sage.rings.species_hyperoctahedral import AtomicHyperoctahedralSpecies
        sage: A = AtomicHyperoctahedralSpecies(2)
        sage: A
        Atomic 2-species
        sage: A.grading_set()
        Integer vectors of length 1

        sage: A = AtomicHyperoctahedralSpecies(2, "X, Y")
        sage: A
        Atomic 2-species in X, Y

    TESTS::

        sage: AtomicHyperoctahedralSpecies(2) is AtomicHyperoctahedralSpecies(ZZ(2))
        True
        sage: AtomicHyperoctahedralSpecies(2) is AtomicHyperoctahedralSpecies(2, "X")
        True
        sage: AtomicHyperoctahedralSpecies(0)
        Traceback (most recent call last):
        ...
        ValueError: r must be a positive integer
    """
    @staticmethod
    def __classcall__(cls, r, names="X"):
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
        names = normalize_names(-1, names)
        return super().__classcall__(cls, r, names)

    def __init__(self, r, names):
        r"""
        Initialize the class of atomic `r`-species.

        EXAMPLES::

            sage: from sage.rings.species_hyperoctahedral import AtomicHyperoctahedralSpecies
            sage: A = AtomicHyperoctahedralSpecies(2)
            sage: A._r
            2
            sage: A._arity
            1

        TESTS:

        We have to exclude ``_test_graded_components``, because
        :meth:`~sage.combinat.integer_vector.IntegerVectors.some_elements`
        yields degrees that are too large::

            sage: TestSuite(AtomicHyperoctahedralSpecies(2)).run(skip="_test_graded_components")
            sage: TestSuite(AtomicHyperoctahedralSpecies(2, "X, Y")).run(skip="_test_graded_components")
        """
        category = SetsWithGrading().Infinite()
        Parent.__init__(self, names=names, category=category)
        self._r = ZZ(r)
        self._arity = len(names)
        self._cache = dict()
        # the grades whose standard species have already been renamed
        self._renamed = set()

    def _repr_(self):
        r"""
        Return a string representation of ``self``.

        EXAMPLES::

            sage: from sage.rings.species_hyperoctahedral import AtomicHyperoctahedralSpecies
            sage: AtomicHyperoctahedralSpecies(3)
            Atomic 3-species
            sage: AtomicHyperoctahedralSpecies(3, "X, Y")
            Atomic 3-species in X, Y
        """
        if len(self._names) == 1:
            return f"Atomic {self._r}-species"
        return f"Atomic {self._r}-species in {', '.join(self._names)}"

    def _an_element_(self):
        r"""
        Return an element of ``self``.

        This is the diagonal cyclic `r`-species: it acts as a full
        `r`-cycle on one block per sort.

        TESTS::

            sage: from sage.rings.species_hyperoctahedral import AtomicHyperoctahedralSpecies
            sage: AtomicHyperoctahedralSpecies(2).an_element()
            X°
            sage: A = AtomicHyperoctahedralSpecies(2, "X, Y")
            sage: A.an_element()
            {((1,2)(3,4),): ({1, 2}, {3, 4})}

            sage: A = AtomicHyperoctahedralSpecies(1, "X, Y")
            sage: A.an_element()
            {((1,2)(3,4),): ({1, 2}, {3, 4})}
        """
        r = self._r
        k = self._arity
        if r == 1:
            # swap two singletons per sort
            gens = [[(2*s + 1, 2*s + 2) for s in range(k)]]
            pi = {s: [2*s + 1, 2*s + 2] for s in range(k)}
        else:
            # rotate one block per sort
            gens = [[tuple(range(r*s + 1, r*s + r + 1)) for s in range(k)]]
            pi = {s: list(range(r*s + 1, r*s + r + 1)) for s in range(k)}
        G = PermutationGroup(gens)
        return self._element_constructor_(G, pi)

    def _rename(self, grade):
        r"""
        Give the standard atomic `r`-species of unit grade their names.

        For a unit grade the atomic `r`-species correspond to the
        divisors `d \mid r`: the stabilizer of the species `C_r / C_d`
        is the cyclic group `C_d` of order `d`.  Following the
        convention of [Henderson2004]_, we display it as ``X@d``, with
        the extremal cases ``X = X@1`` and ``X° = X@r``, and with
        ``X`` for the unique species when `r = 1`.  Every sort receives
        the names obtained from its own name in this way.

        EXAMPLES::

            sage: from sage.rings.species_hyperoctahedral import AtomicHyperoctahedralSpecies
            sage: from sage.groups.perm_gps.hyperoctahedral_group import _wreath_group
            sage: A = AtomicHyperoctahedralSpecies(2)
            sage: A(_wreath_group(2, 1))
            X°
            sage: A = AtomicHyperoctahedralSpecies(4, "X, Y")
            sage: sorted(A.graded_component([0, 1]), key=str)
            [Y, Y@2, Y°]
        """
        if sum(grade) != 1:
            return
        s = list(grade).index(1)
        W = _wreath_group(self._r, 1)
        cycle = W.gens()[0]
        for d in divisors(self._r):
            G = W.subgroup([cycle ** (self._r // d)])
            if d == 1:
                name = self._names[s]
            elif d == self._r:
                name = self._names[s] + "°"
            else:
                name = f"{self._names[s]}@{d}"
            self(G, _canonical_dompart(self._r, grade),
                 check=False).rename(name)

    def _element_constructor_(self, G, pi=None, check=True):
        r"""
        Construct the atomic `r`-species with the given data.

        INPUT:

        - ``G`` -- element of ``self`` (in this case ``pi`` must be
          ``None``) or a permutation group which is a subgroup of
          `W(r,n)` on the standard domain
        - ``pi`` -- a dictionary (or iterable) mapping sorts to
          iterables whose union is the domain of ``G``; if the arity is
          one, ``pi`` can be omitted
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

        TESTS::

            sage: A(G)
            {((1,2), (3,4), (1,3)(2,4))}
        """
        if parent(G) is self:
            if pi is None:
                return G
            raise ValueError("cannot reassign sorts to an atomic species")
        if not isinstance(G, PermutationGroup_generic):
            raise ValueError(f"{G} must be a permutation group")
        if check:
            _check_standard_domain(G, self._r)
            if len(_hyperoctahedral_disjoint_direct_product_decomposition(G, self._r)) != 1:
                raise ValueError(f"{G.gens()} is not directly indecomposable")
        if pi is None:
            if self._arity == 1:
                pi = {0: G.domain()}
            else:
                raise ValueError("the assignment of sorts to the domain elements must be provided")
        elif not isinstance(pi, dict):
            pi = dict(enumerate(pi))
        if check:
            if not set(pi).issubset(range(self._arity)):
                raise ValueError(f"keys of pi (={pi.keys()}) must be in range({self._arity})")
            if (sum(len(p) for p in pi.values()) != len(G.domain())
                    or set(chain.from_iterable(pi.values())) != set(G.domain())):
                raise ValueError(f"values of pi (={pi.values()}) must partition the domain of G (={G.domain()})")
        dompart = [sorted(pi.get(s, [])) for s in range(self._arity)]
        return self.element_class(self, G, dompart)

    def grading_set(self):
        r"""
        Return the grading set of ``self``.

        This is the set of integer vectors whose length is the arity of
        ``self``.

        EXAMPLES::

            sage: from sage.rings.species_hyperoctahedral import AtomicHyperoctahedralSpecies
            sage: AtomicHyperoctahedralSpecies(2).grading_set()
            Integer vectors of length 1
            sage: AtomicHyperoctahedralSpecies(2, "X, Y").grading_set()
            Integer vectors of length 2
        """
        return IntegerVectors(length=self._arity)

    def subset(self, size):
        r"""
        Return the set of atomic `r`-species of total degree ``size``.

        EXAMPLES::

            sage: from sage.rings.species_hyperoctahedral import AtomicHyperoctahedralSpecies
            sage: A = AtomicHyperoctahedralSpecies(2)
            sage: sorted(a.degree() for a in A.subset(2))
            [2, 2, 2, 2, 2]

            sage: A = AtomicHyperoctahedralSpecies(2, "X, Y")
            sage: sorted(A.subset(1), key=str)
            [X, X°, Y, Y°]
        """
        result = Set()
        for grade in IntegerVectors(size, length=self._arity):
            result = result.union(self.graded_component(grade))
        return result

    def graded_component(self, grade):
        r"""
        Return the set of atomic `r`-species of grade ``grade``.

        EXAMPLES::

            sage: from sage.rings.species_hyperoctahedral import AtomicHyperoctahedralSpecies
            sage: A = AtomicHyperoctahedralSpecies(2)
            sage: len(A.graded_component(2))
            5

            sage: A = AtomicHyperoctahedralSpecies(2, "X, Y")
            sage: len(A.graded_component([1, 0]))
            2
            sage: len(A.graded_component([1, 1]))
            1

        The unique atom of grade `(1,1)` for `r = 2` is the diagonal
        `C_2`::

            sage: a = next(iter(A.graded_component([1, 1]))); a
            {((1,2)(3,4),): ({1, 2}, {3, 4})}

        For `r = 1` the number of atoms agrees with the number of
        atomic species::

            sage: from sage.rings.species import AtomicSpecies
            sage: A1 = AtomicHyperoctahedralSpecies(1, "X, Y")
            sage: B = AtomicSpecies("X, Y")
            sage: all(len(A1.graded_component(list(mc))) == len(B.graded_component(list(mc)))
            ....:     for mc in IntegerVectors(3, length=2))
            True
        """
        if not hasattr(grade, '__len__'):
            grade = (grade,)
        if len(grade) != self._arity:
            raise ValueError("invalid degree")
        grade = _check_grade(grade)
        return Set([self(G, _canonical_dompart(self._r, grade), check=False)
                    for G in _wreath_young_subgroup_classes(self._r, grade)
                    if len(_hyperoctahedral_disjoint_direct_product_decomposition(G, self._r)) == 1])

    def __contains__(self, x):
        r"""
        Return whether ``x`` is an atomic `r`-species, or a subgroup
        together with an assignment of sorts defining one.

        EXAMPLES::

            sage: from sage.rings.species_hyperoctahedral import AtomicHyperoctahedralSpecies
            sage: from sage.groups.perm_gps.hyperoctahedral_group import _wreath_group
            sage: A = AtomicHyperoctahedralSpecies(1)
            sage: _wreath_group(1, 1).subgroup([]) in A
            True

        TESTS::

            sage: A = AtomicHyperoctahedralSpecies(2, "X, Y")
            sage: from sage.groups.perm_gps.hyperoctahedral_group import _wreath_young_subgroup
            sage: W = _wreath_young_subgroup(2, [1, 1])
            sage: (W.subgroup([[(1, 2), (3, 4)]]), {0: [1, 2], 1: [3, 4]}) in A
            True
            sage: (W.subgroup([(1, 2), (3, 4)]), {0: [1, 2], 1: [3, 4]}) in A
            False
            sage: (W.subgroup([[(1, 2), (3, 4)]]), {0: [1, 3], 1: [2, 4]}) in A
            False
        """
        if parent(x) is self:
            return True
        if isinstance(x, PermutationGroup_generic):
            if self._arity == 1:
                G = x
                pi = {0: G.domain()}
            else:
                return False
        else:
            G, pi = x
            if not isinstance(G, PermutationGroup_generic):
                return False
            if not isinstance(pi, dict):
                pi = dict(enumerate(pi))
        if not set(pi).issubset(range(self._arity)):
            return False
        if (sum(len(p) for p in pi.values()) != len(G.domain())
                or set(chain.from_iterable(pi.values())) != set(G.domain())):
            return False
        degree = G.degree()
        if degree % self._r:
            return False
        W = _wreath_group(self._r, degree // self._r)
        if set(G.domain()) != set(W.domain()):
            return False
        dompart = [sorted(pi.get(s, [])) for s in range(self._arity)]
        # each sort must be a union of complete C_r-blocks
        for s_points in dompart:
            s_set = set(s_points)
            for b in _wreath_blocks(self._r, degree // self._r):
                if len(s_set.intersection(b)) not in (0, len(b)):
                    return False
        for orbit in G.orbits():
            if not any(set(orbit).issubset(s) for s in dompart):
                return False
        if libgap.IsSubgroup(W.gap(), G.gap()) != True:
            return False
        return len(_hyperoctahedral_disjoint_direct_product_decomposition(G, self._r)) == 1

    class Element(WithEqualityById,
                  Element,
                  WithPicklingByInitArgs,
                  metaclass=InheritComparisonClasscallMetaclass):
        r"""
        An atomic `r`-species.
        """
        @staticmethod
        def __classcall__(cls, parent, G, dompart):
            r"""
            Normalize the input for unique representation.

            The `C_r`-blocks of the domain are relabelled so that the
            blocks of sort `0` come first, then the blocks of sort
            `1`, and so on.  Afterwards the group is replaced by the
            canonical representative of its conjugacy class in the
            ambient wreath Young subgroup and the sort partition is the
            canonical one.

            INPUT:

            - ``G`` -- a directly indecomposable permutation group
            - ``dompart`` -- an iterable of `k` iterables, where `k` is
              the arity, assigning each element of the domain of ``G``
              to a sort

            .. WARNING::

                We do not check whether ``G`` is indeed directly
                indecomposable.

            TESTS::

                sage: from sage.rings.species_hyperoctahedral import AtomicHyperoctahedralSpecies
                sage: from sage.groups.perm_gps.hyperoctahedral_group import _wreath_group
                sage: A = AtomicHyperoctahedralSpecies(2)
                sage: G = _wreath_group(2, 2)
                sage: A(G.subgroup([[(1, 3), (2, 4)]])) is A(G.subgroup([[(1, 3), (2, 4)]]))
                True

            The relabelling of the sorts identifies species which
            differ only by the ordering of the blocks.  Here the
            diagonal `C_2` is the unique atom of grade `(1,1)`::

                sage: from sage.groups.perm_gps.hyperoctahedral_group import _wreath_young_subgroup
                sage: A = AtomicHyperoctahedralSpecies(2, "X, Y")
                sage: W = _wreath_young_subgroup(2, [1, 1])
                sage: a = A(W.subgroup([[(1, 2), (3, 4)]]), {0: [1, 2], 1: [3, 4]})
                sage: a is A(W.subgroup([[(1, 2), (3, 4)]]), {0: [3, 4], 1: [1, 2]})
                True
                sage: a is A(W.subgroup([[(1, 2), (3, 4)]]), {0: [2, 1], 1: [4, 3]})
                True

            The subgroup acting on one sort only is not atomic::

                sage: a is A(W.subgroup([(1, 2)]), {0: [1, 2], 1: [3, 4]})
                Traceback (most recent call last):
                ...
                ValueError: ((1,2),) is not directly indecomposable
            """
            r = parent._r
            _check_standard_domain(G, r)

            dompart = [sorted(b) for b in dompart]

            # each sort must be a union of complete C_r-blocks
            point_sort = {}
            for s, b in enumerate(dompart):
                point_sort.update({p: s for p in b})
            for blk in _wreath_blocks(r, G.degree() // r):
                s = point_sort.get(blk[0])
                for p in blk:
                    if point_sort.get(p) != s:
                        raise ValueError(f"the assignment of sorts {dompart} "
                                         f"must be a union of C_{r}-blocks")

            # every orbit of G must be contained in a single sort
            for orbit in G.orbits():
                if not any(set(orbit).issubset(b) for b in dompart):
                    raise ValueError(f"all elements of orbit {list(orbit)} "
                                     f"must have the same sort")

            grade = tuple(len(b) // r for b in dompart)

            # relabel the blocks so that the blocks of sort 0 come
            # first, then the blocks of sort 1, and so on
            ordered = list(chain.from_iterable(dompart))
            if ordered != list(range(1, G.degree() + 1)):
                relabel = {p: i + 1 for i, p in enumerate(ordered)}
                gens = []
                for gen in G.gens():
                    cycles = [tuple(relabel[p] for p in cyc)
                              for cyc in gen.cycle_tuples()]
                    cycles = [cyc for cyc in cycles if cyc]
                    if cycles:
                        gens.append(PermutationGroupElement(cycles))
                G = PermutationGroup(gens, domain=range(1, G.degree() + 1))

            idx, rep = _canonical_wreath_subgroup_index(G, r, grade)
            key = (grade, idx)
            if key in parent._cache:
                return parent._cache[key]
            elm = WithPicklingByInitArgs.__classcall__(
                cls, parent, rep, _canonical_dompart(r, grade))
            parent._cache[key] = elm
            return elm

        def __init__(self, parent, dis, domain_partition):
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

                sage: A = AtomicHyperoctahedralSpecies(2, "X, Y")
                sage: from sage.groups.perm_gps.hyperoctahedral_group import _wreath_young_subgroup
                sage: W = _wreath_young_subgroup(2, [1, 1])
                sage: a = A(W.subgroup([[(1, 2), (3, 4)]]), {0: [1, 2], 1: [3, 4]})
                sage: TestSuite(a).run()
                sage: loads(dumps(a)) is a
                True
            """
            Element.__init__(self, parent)
            self._dis = dis
            self._dompart = domain_partition
            self._mc = tuple(len(v) // parent._r for v in self._dompart)
            self._tc = sum(self._mc)

        def _repr_(self):
            r"""
            Return a string representation of ``self``.

            The first time a grade is displayed, the standard names for
            that grade are installed by :meth:`_rename`.

            EXAMPLES::

                sage: from sage.rings.species_hyperoctahedral import AtomicHyperoctahedralSpecies
                sage: from sage.groups.perm_gps.hyperoctahedral_group import _wreath_group
                sage: A = AtomicHyperoctahedralSpecies(2)
                sage: A(_wreath_group(2, 2).subgroup([[(1, 3), (2, 4)]]))
                {((1,3)(2,4),)}

                sage: A = AtomicHyperoctahedralSpecies(2, "X, Y")
                sage: from sage.groups.perm_gps.hyperoctahedral_group import _wreath_young_subgroup
                sage: W = _wreath_young_subgroup(2, [1, 1])
                sage: A(W.subgroup([[(1, 2), (3, 4)]]), {0: [1, 2], 1: [3, 4]})
                {((1,2)(3,4),): ({1, 2}, {3, 4})}
            """
            P = self.parent()
            grade = tuple(self._mc)
            if grade not in P._renamed:
                P._renamed.add(grade)
                P._rename(grade)
                return repr(self)
            if P._arity == 1:
                return "{" + f"{self._dis.gens()}" + "}"
            dompart = ', '.join("{" + repr(sorted(b))[1:-1] + "}"
                                for b in self._dompart)
            return "{" + f"{self._dis.gens()}: ({dompart})" + "}"

        def permutation_group(self):
            r"""
            Return the permutation group representing ``self``, together
            with the partition of its domain into sorts.

            The group acts on the standard domain and is the canonical
            representative of its conjugacy class in the ambient wreath
            Young subgroup.  The domain partition consists of complete
            `C_r`-blocks.

            EXAMPLES::

                sage: from sage.rings.species_hyperoctahedral import AtomicHyperoctahedralSpecies
                sage: from sage.groups.perm_gps.hyperoctahedral_group import _wreath_group
                sage: A = AtomicHyperoctahedralSpecies(2)
                sage: A(_wreath_group(2, 1)).permutation_group()
                (Subgroup generated by [(1,2)] of (Permutation Group with generators [(1,2)]), (frozenset({1, 2}),))

                sage: A = AtomicHyperoctahedralSpecies(2, "X, Y")
                sage: from sage.groups.perm_gps.hyperoctahedral_group import _wreath_young_subgroup
                sage: W = _wreath_young_subgroup(2, [1, 1])
                sage: A(W.subgroup([[(1, 2), (3, 4)]]), {0: [1, 2], 1: [3, 4]}).permutation_group()
                (Subgroup generated by [(1,2)(3,4)] of (Permutation Group with generators [(3,4), (1,2)]), (frozenset({1, 2}), frozenset({3, 4})))
            """
            return self._dis, self._dompart

        def degree(self):
            r"""
            Return the degree of ``self``.

            This is the total number of `C_r`-blocks in its domain.

            EXAMPLES::

                sage: from sage.rings.species_hyperoctahedral import AtomicHyperoctahedralSpecies
                sage: from sage.groups.perm_gps.hyperoctahedral_group import _wreath_group
                sage: A = AtomicHyperoctahedralSpecies(2)
                sage: A(_wreath_group(2, 2).subgroup([[(1, 3), (2, 4)]])).degree()
                2
            """
            return self._tc

        def grade(self):
            r"""
            Return the grade of ``self``.

            This is the number of `C_r`-blocks in each sort.

            EXAMPLES::

                sage: from sage.rings.species_hyperoctahedral import AtomicHyperoctahedralSpecies
                sage: from sage.groups.perm_gps.hyperoctahedral_group import _wreath_group
                sage: A = AtomicHyperoctahedralSpecies(2)
                sage: A(_wreath_group(2, 2).subgroup([[(1, 3), (2, 4)]])).grade()
                [2]

                sage: A = AtomicHyperoctahedralSpecies(2, "X, Y")
                sage: from sage.groups.perm_gps.hyperoctahedral_group import _wreath_young_subgroup
                sage: W = _wreath_young_subgroup(2, [1, 1])
                sage: A(W.subgroup([[(1, 2), (3, 4)]]), {0: [1, 2], 1: [3, 4]}).grade()
                [1, 1]
            """
            return self.parent().grading_set()(list(self._mc))

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


class MolecularHyperoctahedralSpecies(IndexedFreeAbelianMonoid):
    r"""
    The monoid of molecular `r`-species.

    This is the commutative free abelian monoid generated by the atomic
    `r`-species, exactly as
    :class:`~sage.rings.species.MolecularSpecies` is generated by
    :class:`~sage.rings.species.AtomicSpecies`.

    INPUT:

    - ``r`` -- positive integer; the order of the cyclic group
    - ``names`` -- an iterable of strings for the sorts of the species
      (default: ``"X"``)

    EXAMPLES::

        sage: from sage.rings.species_hyperoctahedral import MolecularHyperoctahedralSpecies
        sage: M = MolecularHyperoctahedralSpecies(2)
        sage: M
        Molecular 2-species

        sage: M = MolecularHyperoctahedralSpecies(2, "X, Y")
        sage: M
        Molecular 2-species in X, Y

    TESTS::

        sage: from sage.rings.species_hyperoctahedral import MolecularHyperoctahedralSpecies
        sage: MolecularHyperoctahedralSpecies(2) is MolecularHyperoctahedralSpecies(ZZ(2))
        True
        sage: MolecularHyperoctahedralSpecies(2) is MolecularHyperoctahedralSpecies(2, "X")
        True
    """
    @staticmethod
    def __classcall__(cls, r, names="X"):
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
        names = normalize_names(-1, names)
        return UniqueRepresentation.__classcall__(cls, r, names)

    def __init__(self, r, names):
        r"""
        Initialize the monoid of molecular `r`-species.

        EXAMPLES::

            sage: from sage.rings.species_hyperoctahedral import MolecularHyperoctahedralSpecies
            sage: M = MolecularHyperoctahedralSpecies(2)
            sage: M._r
            2
            sage: M._arity
            1

        TESTS:

        We have to exclude ``_test_graded_components``, because
        :meth:`~sage.combinat.integer_vector.IntegerVectors.some_elements`
        yields degrees that are too large::

            sage: TestSuite(MolecularHyperoctahedralSpecies(2)).run(skip="_test_graded_components")
            sage: TestSuite(MolecularHyperoctahedralSpecies(2, "X, Y")).run(skip="_test_graded_components")
        """
        indices = AtomicHyperoctahedralSpecies(r, names)
        category = Monoids().Commutative() & SetsWithGrading().Infinite()
        IndexedFreeAbelianMonoid.__init__(self, indices, prefix='',
                                          bracket=False, category=category)
        self._r = ZZ(r)
        self._arity = indices._arity

    def _repr_(self):
        r"""
        Return a string representation of ``self``.

        EXAMPLES::

            sage: from sage.rings.species_hyperoctahedral import MolecularHyperoctahedralSpecies
            sage: MolecularHyperoctahedralSpecies(3)
            Molecular 3-species
            sage: MolecularHyperoctahedralSpecies(3, "X, Y")
            Molecular 3-species in X, Y
        """
        if len(self._indices._names) == 1:
            return f"Molecular {self._r}-species"
        return f"Molecular {self._r}-species in {', '.join(self._indices._names)}"

    def _first_ngens(self, n):
        r"""
        Used by the preparser for ``F.<x> = ...``.

        We do not use the generic implementation of
        :class:`sage.monoids.indexed_free_monoid.IndexedFreeAbelianMonoid`,
        because the atomic species cannot be enumerated.

        EXAMPLES::

            sage: from sage.rings.species_hyperoctahedral import MolecularHyperoctahedralSpecies
            sage: M = MolecularHyperoctahedralSpecies(2, "X, Y")
            sage: M._first_ngens(1)
            (X,)
            sage: M._first_ngens(2)
            (X, Y)
        """
        singletons = [sorted(self._indices.graded_component(grade), key=str)[0]
                      for grade in IntegerVectors(1, length=self._arity)]
        return tuple(self.gen(a) for a in singletons[:n])

    def _element_constructor_(self, G, pi=None, check=True):
        r"""
        Construct the molecular `r`-species with the given data.

        INPUT:

        - ``G`` -- one of the following:

          - an element of ``self``
          - a permutation group which is a subgroup of `W(r,n)` on the
            standard domain
          - a dictionary from atoms to exponents
          - a triple ``(X, a, side)`` consisting of a finite set, an
            action and a string ``'left'`` or ``'right'``; the side can
            be omitted, it is then assumed to be ``'right'``

        - ``pi`` -- a dictionary (or iterable) mapping sorts to iterables
          whose union is the domain of ``G``, resp. the domain of the
          acting wreath product if ``G`` is an action; if the arity is
          one and ``G`` is a permutation group, ``pi`` can be omitted
        - ``check`` -- boolean (default: ``True``); whether to check
          the dictionary input and the group action

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

        The stabilizer of a point under the natural action of
        `W(2,2)`::

            sage: W = _wreath_group(2, 2)
            sage: X = list(W.domain())
            sage: a = lambda g, x: g(x)
            sage: M((X, a, 'left'), {0: list(W.domain())})
            X*X°

        For several sorts, each `C_r`-block of the domain is assigned to
        a sort.  The trivial subgroup of `W(2;1,1)` decomposes into the
        product of the trivial species in each sort::

            sage: from sage.groups.perm_gps.hyperoctahedral_group import _wreath_young_subgroup
            sage: M = MolecularHyperoctahedralSpecies(2, "X, Y")
            sage: W = _wreath_young_subgroup(2, [1, 1])
            sage: M(W.subgroup([]), {0: [1, 2], 1: [3, 4]})
            X*Y
            sage: M(W.subgroup([(1, 2), (3, 4)]), {0: [1, 2], 1: [3, 4]})
            X°*Y°

        The diagonal `C_2` is directly indecomposable, whence atomic::

            sage: M(W.subgroup([[(1, 2), (3, 4)]]), {0: [1, 2], 1: [3, 4]})
            {((1,2)(3,4),): ({1, 2}, {3, 4})}

        TESTS::

            sage: TestSuite(M(W.subgroup([(1, 2), (3, 4)]), {0: [1, 2], 1: [3, 4]})).run()
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
        if isinstance(G, tuple):
            if len(G) == 2:
                X, a = G
                side = 'right'
            else:
                X, a, side = G
                if side not in ['left', 'right']:
                    raise ValueError(f"the side must be 'right' or 'left', but is {side}")
            stabilizers = _wreath_stabilizers(X, a, side, pi, self._r,
                                              self._arity, check=check)
            if len(stabilizers) > 1:
                raise ValueError("action is not transitive")
            G, pi = stabilizers[0]
        if not isinstance(G, PermutationGroup_generic):
            raise ValueError(f"{G} must be a permutation group")
        _check_standard_domain(G, self._r)
        if pi is None:
            if self._arity == 1:
                pi = {0: G.domain()}
            elif G.degree() == 0:
                pi = {}
            else:
                raise ValueError("the assignment of sorts to the domain elements must be provided")
        elif not isinstance(pi, dict):
            pi = dict(enumerate(pi))
        if not set(pi).issubset(range(self._arity)):
            raise ValueError(f"keys of pi (={pi.keys()}) must be in range({self._arity})")
        if (sum(len(p) for p in pi.values()) != len(G.domain())
                or set(chain.from_iterable(pi.values())) != set(G.domain())):
            raise ValueError(f"values of pi (={pi.values()}) must partition the domain of G (={G.domain()})")
        dompart = [sorted(pi.get(s, [])) for s in range(self._arity)]

        decomposition = _hyperoctahedral_disjoint_direct_product_decomposition(G, self._r)
        result = self.one()
        for comp in decomposition:
            comp = sorted(comp)
            comp_set = set(comp)
            H = _standardize_component(G, comp, self._r)
            relabel = {p: i + 1 for i, p in enumerate(comp)}
            pi_H = {}
            for s in range(self._arity):
                pts = [relabel[p] for p in dompart[s] if p in comp_set]
                if pts:
                    pi_H[s] = pts
            result *= self.gen(self._indices(H, pi_H))
        return result

    def grading_set(self):
        r"""
        Return the grading set of ``self``.

        This is the set of integer vectors whose length is the arity of
        ``self``.

        EXAMPLES::

            sage: from sage.rings.species_hyperoctahedral import MolecularHyperoctahedralSpecies
            sage: MolecularHyperoctahedralSpecies(2).grading_set()
            Integer vectors of length 1
            sage: MolecularHyperoctahedralSpecies(2, "X, Y").grading_set()
            Integer vectors of length 2
        """
        return IntegerVectors(length=self._arity)

    def subset(self, size):
        r"""
        Return the set of molecular `r`-species of total degree ``size``.

        EXAMPLES::

            sage: from sage.rings.species_hyperoctahedral import MolecularHyperoctahedralSpecies
            sage: M = MolecularHyperoctahedralSpecies(2)
            sage: len(M.subset(2))
            8

            sage: M = MolecularHyperoctahedralSpecies(2, "X, Y")
            sage: len(M.subset(2))
            21
        """
        result = Set()
        for grade in IntegerVectors(size, length=self._arity):
            result = result.union(self.graded_component(grade))
        return result

    def graded_component(self, grade):
        r"""
        Return the set of molecular `r`-species of grade ``grade``.

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

        For several sorts the ambient group is the wreath Young
        subgroup `W(r;n_1,\ldots,n_k)`::

            sage: M = MolecularHyperoctahedralSpecies(2, "X, Y")
            sage: len(M.graded_component([1, 1]))
            5

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
        if not hasattr(grade, '__len__'):
            grade = (grade,)
        if len(grade) != self._arity:
            raise ValueError("invalid degree")
        grade = _check_grade(grade)
        return Set([self(rep, _canonical_dompart(self._r, grade))
                    for rep in _wreath_young_subgroup_classes(self._r, grade)])

    def _type1_substitute_molecular(self, M, molecules):
        r"""
        Substitute `r`-molecular species into an ordinary molecular species.

        This is the final, group-theoretic step of the first Henderson
        substitution: the ordinary molecular species ``M`` is an
        abstract coloured species whose sorts correspond to the
        entries of ``molecules``, and every ordinary point is
        replaced by the corresponding substituted `r`-molecule.

        INPUT:

        - ``M`` -- a molecular species of an ordinary
          :class:`~sage.rings.species.PolynomialSpecies`;
          its ``permutation_group()`` supplies the ordinary group
          `H` and the partition of its points into sorts
        - ``molecules`` -- a list of molecular `r`-species, one for
          each sort of ``M``

        OUTPUT:

        An element of ``self``.

        EXAMPLES::

            sage: from sage.rings.species import PolynomialSpecies
            sage: from sage.rings.species_hyperoctahedral import MolecularHyperoctahedralSpecies
            sage: from sage.groups.perm_gps.hyperoctahedral_group import _wreath_group
            sage: P = PolynomialSpecies(QQ, "A, B")
            sage: M = MolecularHyperoctahedralSpecies(2)
            sage: X = M(_wreath_group(2, 1).subgroup([]))
            sage: Xo = M(_wreath_group(2, 1))
            sage: G = PermutationGroup([(1, 2)], domain=[1, 2, 3])
            sage: E2AB = P(G, {0: [1, 2], 1: [3]}).support()[0]
            sage: M._type1_substitute_molecular(E2AB, [Xo, X])
            X*{((1,2), (3,4), (1,3)(2,4))}

        TESTS:

        The group of `E_2(X^\circ)` is the full wreath product::

            sage: P = PolynomialSpecies(QQ, "X")
            sage: M = MolecularHyperoctahedralSpecies(2)
            sage: Xo = M(_wreath_group(2, 1))
            sage: E2 = P(SymmetricGroup(2)).support()[0]
            sage: G = M._type1_substitute_molecular(E2, [Xo]).permutation_group()[0]
            sage: G.order()
            8
        """
        r = self._r
        H, dompart = M.permutation_group()
        n = H.degree()

        # the sort of each ordinary point
        sort_of = {}
        for i, block in enumerate(dompart):
            for j in block:
                sort_of[j] = i

        # sizes (in C_r-blocks) and offsets of the chunks
        sizes = {}
        offsets = {}
        offset = 0
        for j in range(1, n + 1):
            sizes[j] = molecules[sort_of[j]].degree()
            offsets[j] = offset
            offset += sizes[j]
        N = offset

        gens = []
        # Lift the generators of the ordinary outer group to the chunks.
        for h in H.gens():
            perm = list(range(1, N * r + 1))
            for j in range(1, n + 1):
                jp = h(j)
                for p in range(sizes[j] * r):
                    perm[offsets[j] * r + p] = offsets[jp] * r + p + 1
            gens.append(PermutationGroupElement(perm))

        # Embed the generators of each substituted molecule into
        # every occurrence of its sort.
        for i, block in enumerate(dompart):
            K, _ = molecules[i].permutation_group()
            for j in block:
                shift = offsets[j] * r
                for gen in K.gens():
                    cycles = [tuple(shift + p for p in cyc)
                              for cyc in gen.cycle_tuples()]
                    cycles = [cyc for cyc in cycles if cyc]
                    if cycles:
                        gens.append(PermutationGroupElement(cycles))

        W = PermutationGroup(gens, domain=range(1, N * r + 1))
        return self(W)

    class Element(IndexedFreeAbelianMonoidElement):
        r"""
        A molecular `r`-species.
        """
        @cached_method
        def grade(self):
            r"""
            Return the grade of ``self``.

            This is the vector counting the `C_r`-blocks in each
            sort.

            EXAMPLES::

                sage: from sage.rings.species_hyperoctahedral import MolecularHyperoctahedralSpecies
                sage: from sage.groups.perm_gps.hyperoctahedral_group import _wreath_group
                sage: M = MolecularHyperoctahedralSpecies(2)
                sage: M(_wreath_group(2, 2).subgroup([(1, 2), (3, 4)])).grade()
                [2]

                sage: M = MolecularHyperoctahedralSpecies(2, "X, Y")
                sage: from sage.groups.perm_gps.hyperoctahedral_group import _wreath_young_subgroup
                sage: W = _wreath_young_subgroup(2, [1, 1])
                sage: M(W.subgroup([]), {0: [1, 2], 1: [3, 4]}).grade()
                [1, 1]
            """
            P = self.parent()
            S = P.grading_set()
            mons = self._monomial
            if not mons:
                return S([0] * P._arity)
            mc = [sum(n * a._mc[s] for a, n in mons.items())
                  for s in range(P._arity)]
            return S(mc)

        def degree(self):
            r"""
            Return the degree of ``self``.

            This is the total number of `C_r`-blocks in its domain.

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

        @cached_method
        def permutation_group(self):
            r"""
            Return a permutation group representing ``self``, together
            with the partition of its domain into sorts.

            The result acts on the standard domain `\{1, \ldots, rn\}`
            and is the direct product of the permutation groups of the
            atomic factors of ``self``.  The points of each sort of
            each factor are appended after the points of the same sort
            contributed by previous factors.

            EXAMPLES::

                sage: from sage.rings.species_hyperoctahedral import MolecularHyperoctahedralSpecies
                sage: from sage.groups.perm_gps.hyperoctahedral_group import _wreath_group
                sage: M = MolecularHyperoctahedralSpecies(2)
                sage: M(_wreath_group(2, 2).subgroup([(1, 2), (3, 4)])).permutation_group()
                (Permutation Group with generators [(3,4), (1,2)], (frozenset({1, 2, 3, 4}),))
                sage: M.one().permutation_group()
                (Permutation Group with generators [()], (frozenset(),))

            For several sorts the domain partition records which points
            belong to which sort::

                sage: M = MolecularHyperoctahedralSpecies(2, "X, Y")
                sage: from sage.groups.perm_gps.hyperoctahedral_group import _wreath_young_subgroup
                sage: W = _wreath_young_subgroup(2, [1, 1])
                sage: G, dompart = M(W.subgroup([(1, 2)]), {0: [1, 2], 1: [3, 4]}).permutation_group()
                sage: G.order()
                2
                sage: dompart
                (frozenset({1, 2}), frozenset({3, 4}))
            """
            P = self.parent()
            r = P._r
            arity = P._arity
            offset = [0] * arity
            grade = [0] * arity
            gens = []
            for A, e in self._monomial.items():
                H = A.permutation_group()[0]
                for _ in range(e):
                    relabel = {}
                    for s in range(arity):
                        points = sorted(A._dompart[s])
                        for i, p in enumerate(points):
                            relabel[p] = offset[s] + i + 1
                    for gen in H.gens():
                        cycles = [tuple(relabel[p] for p in cyc)
                                  for cyc in gen.cycle_tuples()]
                        cycles = [cyc for cyc in cycles if cyc]
                        if cycles:
                            gens.append(PermutationGroupElement(cycles))
                    for s in range(arity):
                        offset[s] += len(A._dompart[s])
                        grade[s] += A._mc[s]
            dompart = _canonical_dompart(r, tuple(grade))
            return PermutationGroup(gens, domain=range(1, sum(offset) + 1)), dompart


class PolynomialHyperoctahedralSpecies(CombinatorialFreeModule):
    r"""
    The ring of polynomial `r`-species.

    This is the commutative graded algebra over ``base_ring`` whose
    basis is given by the molecular `r`-species, exactly as
    :class:`~sage.rings.species.PolynomialSpecies` is the graded algebra
    whose basis is given by the molecular species.

    INPUT:

    - ``base_ring`` -- a ring
    - ``r`` -- positive integer; the order of the cyclic group
    - ``names`` -- an iterable of strings for the sorts of the
      species (default: ``"X"``)

    EXAMPLES::

        sage: from sage.rings.species_hyperoctahedral import PolynomialHyperoctahedralSpecies
        sage: from sage.groups.perm_gps.hyperoctahedral_group import _wreath_group
        sage: P = PolynomialHyperoctahedralSpecies(QQ, 2)
        sage: P
        Polynomial 2-species over Rational Field
        sage: W = _wreath_group(2, 1)
        sage: P(W.subgroup([])) * P(W)
        X*X°

    The product is associative, commutative and distributive::

        sage: P = PolynomialHyperoctahedralSpecies(QQ, 2)
        sage: W1 = _wreath_group(2, 1)
        sage: W2 = _wreath_group(2, 2)
        sage: a = P(W1)
        sage: b = P(W2.subgroup([]))
        sage: c = P(W2.subgroup([[(1, 3), (2, 4)]]))
        sage: a * b == b * a
        True
        sage: (a * b) * c == a * (b * c)
        True
        sage: a * (b + c) == a * b + a * c
        True
        sage: (a * b).degree()
        3

    For several sorts the generators are the degree one species in
    each sort::

        sage: P.<X, Y> = PolynomialHyperoctahedralSpecies(QQ, 2)
        sage: W = _wreath_group(2, 1)
        sage: P(W.subgroup([]), {0: [1, 2]}) * P(W, {1: [1, 2]})
        X*Y°

    TESTS::

        sage: from sage.rings.species_hyperoctahedral import PolynomialHyperoctahedralSpecies
        sage: PolynomialHyperoctahedralSpecies(QQ, 2) is PolynomialHyperoctahedralSpecies(QQ, ZZ(2))
        True
        sage: PolynomialHyperoctahedralSpecies(QQ, 2) is PolynomialHyperoctahedralSpecies(QQ, ZZ(2), "X")
        True
        sage: PolynomialHyperoctahedralSpecies(QQ, 0)
        Traceback (most recent call last):
        ...
        ValueError: r must be a positive integer
    """
    @staticmethod
    def __classcall__(cls, base_ring, r, names="X"):
        r"""
        Normalize the arguments for unique representation.

        TESTS::

            sage: from sage.rings.species_hyperoctahedral import PolynomialHyperoctahedralSpecies
            sage: PolynomialHyperoctahedralSpecies(QQ, 2) is PolynomialHyperoctahedralSpecies(QQ, 2)
            True
        """
        r = ZZ(r)
        if r < 1:
            raise ValueError("r must be a positive integer")
        names = normalize_names(-1, names)
        return super().__classcall__(cls, base_ring, r, names)

    def __init__(self, base_ring, r, names):
        r"""
        Initialize the ring of polynomial `r`-species.

        TESTS::

            sage: from sage.rings.species_hyperoctahedral import PolynomialHyperoctahedralSpecies
            sage: P = PolynomialHyperoctahedralSpecies(ZZ, 2)
            sage: TestSuite(P).run()
            sage: P2 = PolynomialHyperoctahedralSpecies(ZZ, 2, "X, Y")
            sage: TestSuite(P2).run()
        """
        self._r = ZZ(r)
        self._arity = len(names)
        category = GradedAlgebrasWithBasis(base_ring).Commutative()
        CombinatorialFreeModule.__init__(self, base_ring,
                                         basis_keys=MolecularHyperoctahedralSpecies(r, names),
                                         category=category,
                                         prefix='', bracket=False)

    def _repr_(self):
        r"""
        Return a string representation of ``self``.

        EXAMPLES::

            sage: from sage.rings.species_hyperoctahedral import PolynomialHyperoctahedralSpecies
            sage: PolynomialHyperoctahedralSpecies(ZZ, 3)
            Polynomial 3-species over Integer Ring
            sage: PolynomialHyperoctahedralSpecies(ZZ, 3, "X, Y")
            Polynomial 3-species in X, Y over Integer Ring
        """
        if self._arity == 1:
            return f"Polynomial {self._r}-species over {self.base_ring()}"
        names = self._indices._indices._names
        return (f"Polynomial {self._r}-species in {', '.join(names)} "
                f"over {self.base_ring()}")

    def _first_ngens(self, n):
        r"""
        Used by the preparser for ``F.<x> = ...``.

        We do not use the generic implementation of
        :class:`sage.combinat.CombinatorialFreeModule`, because we do
        not want to implement `gens`.

        EXAMPLES::

            sage: from sage.rings.species_hyperoctahedral import PolynomialHyperoctahedralSpecies
            sage: P.<X, Y> = PolynomialHyperoctahedralSpecies(QQ, 2)  # indirect doctest
            sage: X.degree()
            1
            sage: X + 2*Y
            X + 2*Y

        Only the first ``n`` degree one species are returned::

            sage: P = PolynomialHyperoctahedralSpecies(QQ, 2, "X, Y")
            sage: P._first_ngens(1)
            (X,)
            sage: P._first_ngens(2)
            (X, Y)
        """
        singletons = [sorted(self._indices._indices.graded_component(grade), key=str)[0]
                      for grade in IntegerVectors(1, length=self._arity)]
        return tuple(self(a) for a in singletons[:n])

    def _element_constructor_(self, G, pi=None, check=True):
        r"""
        Construct the polynomial `r`-species with the given data.

        INPUT:

        - ``G`` -- one of the following:

          - an element of ``self``
          - an atomic or molecular `r`-species
          - a permutation group which is a subgroup of `W(r,n)` on the
            standard domain
          - a dictionary from molecular `r`-species to elements of the
            base ring
          - a triple ``(X, a, side)`` consisting of a finite set, an
            action and a string ``'left'`` or ``'right'``; the side can
            be omitted, it is then assumed to be ``'right'``

        - ``pi`` -- a dictionary (or iterable) mapping sorts to
          iterables whose union is the domain of ``G``, resp. the
          domain of the acting wreath product if ``G`` is an action;
          if the arity is one and ``G`` is a permutation group, ``pi``
          can be omitted
        - ``check`` -- boolean (default: ``True``); whether to check
          the dictionary input and the group action

        EXAMPLES::

            sage: from sage.rings.species_hyperoctahedral import (
            ....:     PolynomialHyperoctahedralSpecies, MolecularHyperoctahedralSpecies)
            sage: from sage.groups.perm_gps.hyperoctahedral_group import _wreath_group
            sage: P = PolynomialHyperoctahedralSpecies(QQ, 2)
            sage: M = MolecularHyperoctahedralSpecies(2)
            sage: W = _wreath_group(2, 2)
            sage: P(M(W.subgroup([])))
            X^2
            sage: P(W.subgroup([(1, 2), (3, 4)]))
            X°^2

        A subgroup which is not directly indecomposable is decomposed
        into molecular factors::

            sage: P(W.subgroup([(1, 2)]))
            X°*X

        The stabilizer of a point under the natural action of
        `W(2,2)`::

            sage: X = list(W.domain())
            sage: a = lambda g, x: g(x)
            sage: P((X, a, 'left'), {0: list(W.domain())})
            X*X°

        For several sorts the assignment of the `C_r`-blocks to sorts
        must be provided::

            sage: from sage.groups.perm_gps.hyperoctahedral_group import _wreath_young_subgroup
            sage: P = PolynomialHyperoctahedralSpecies(QQ, 2, "X, Y")
            sage: W = _wreath_young_subgroup(2, [1, 1])
            sage: P(W.subgroup([[(1, 2), (3, 4)]]), {0: [1, 2], 1: [3, 4]})
            {((1,2)(3,4),): ({1, 2}, {3, 4})}
            sage: P(W.subgroup([]), {0: [1, 2], 1: [3, 4]})
            X*Y

        TESTS::

            sage: P = PolynomialHyperoctahedralSpecies(QQ, 2)
            sage: f = P.one()
            sage: P(f) is f
            True
        """
        if parent(G) is self:
            raise ValueError("cannot reassign data to a polynomial species")

        if isinstance(G, AtomicHyperoctahedralSpecies.Element):
            G = self._indices({G: ZZ.one()})

        if isinstance(G, MolecularHyperoctahedralSpecies.Element):
            if check and G.parent() is not self._indices:
                raise ValueError(f"{G} must be a {self._indices}")
            return self._from_dict({G: self.base_ring().one()})

        if isinstance(G, dict):
            if check:
                if not all(M.parent() is self._indices for M in G):
                    raise ValueError(f"all keys of the dict {G} must be {self._indices}")
                if not all(e in self.base_ring() for e in G.values()):
                    raise ValueError(f"all values of the dict {G} must be in {self.base_ring()}")
            return self._from_dict(G)

        if isinstance(G, tuple):
            if len(G) == 2:
                X, a = G
                side = 'right'
            else:
                X, a, side = G
                if side not in ['left', 'right']:
                    raise ValueError(f"the side must be 'right' or 'left', but is {side}")
            stabilizers = _wreath_stabilizers(X, a, side, pi, self._r,
                                              self._arity, check=check)
            result = self.zero()
            for H, pi_H in stabilizers:
                result += self.monomial(self._indices(H, pi_H))
            return result

        if isinstance(G, PermutationGroup_generic):
            M = self._indices(G, pi)
            return self._from_dict({M: ZZ.one()})

        raise ValueError(f"{G} must be an element of the base ring, a permutation "
                         "group or a molecular species")

    def change_ring(self, R):
        r"""
        Return the base change of ``self`` to `R`.

        EXAMPLES::

            sage: from sage.rings.species_hyperoctahedral import PolynomialHyperoctahedralSpecies
            sage: P = PolynomialHyperoctahedralSpecies(ZZ, 2)
            sage: P.change_ring(QQ)
            Polynomial 2-species over Rational Field
            sage: P.change_ring(ZZ) is P
            True
            sage: P2 = PolynomialHyperoctahedralSpecies(ZZ, 2, "X, Y")
            sage: P2.change_ring(QQ)
            Polynomial 2-species in X, Y over Rational Field
        """
        if R is self.base_ring():
            return self
        return PolynomialHyperoctahedralSpecies(R, self._r,
                                                self._indices._indices._names)

    def degree_on_basis(self, m):
        r"""
        Return the degree of the molecular `r`-species ``m``.

        EXAMPLES::

            sage: from sage.rings.species_hyperoctahedral import PolynomialHyperoctahedralSpecies
            sage: from sage.groups.perm_gps.hyperoctahedral_group import _wreath_group
            sage: P = PolynomialHyperoctahedralSpecies(QQ, 2)
            sage: P.degree_on_basis(P(_wreath_group(2, 3)).support()[0])
            3
        """
        return m.degree()

    @cached_method
    def one_basis(self):
        r"""
        Return the index of the multiplicative unit.

        EXAMPLES::

            sage: from sage.rings.species_hyperoctahedral import PolynomialHyperoctahedralSpecies
            sage: PolynomialHyperoctahedralSpecies(QQ, 2).one_basis()
            1
        """
        return self._indices.one()

    def product_on_basis(self, H, K):
        r"""
        Return the product of the basis elements indexed by ``H`` and ``K``.

        The product of two molecular `r`-species is their direct product
        with disjoint supports, which is the product in the free
        commutative monoid of molecular `r`-species.

        EXAMPLES::

            sage: from sage.rings.species_hyperoctahedral import PolynomialHyperoctahedralSpecies
            sage: from sage.groups.perm_gps.hyperoctahedral_group import _wreath_group
            sage: P = PolynomialHyperoctahedralSpecies(QQ, 2)
            sage: M = P._indices
            sage: H = M(_wreath_group(2, 1).subgroup([]))
            sage: K = M(_wreath_group(2, 1))
            sage: P.product_on_basis(H, K)
            X*X°
        """
        return self.element_class(self, {H * K: ZZ.one()})

    class Element(CombinatorialFreeModule.Element):
        r"""
        A (virtual) polynomial `r`-species.
        """
        def is_constant(self):
            r"""
            Return whether ``self`` is a constant polynomial `r`-species.

            EXAMPLES::

                sage: from sage.rings.species_hyperoctahedral import PolynomialHyperoctahedralSpecies
                sage: from sage.groups.perm_gps.hyperoctahedral_group import _wreath_group
                sage: P = PolynomialHyperoctahedralSpecies(QQ, 2)
                sage: X = P(_wreath_group(2, 1).subgroup([]))
                sage: X.is_constant()
                False
                sage: (3*P.one()).is_constant()
                True
                sage: P(0).is_constant()
                True
                sage: (1 + X).is_constant()
                False
            """
            return self.is_zero() or not self.maximal_degree()

        def is_virtual(self):
            r"""
            Return whether ``self`` is a virtual polynomial `r`-species.

            EXAMPLES::

                sage: from sage.rings.species_hyperoctahedral import PolynomialHyperoctahedralSpecies
                sage: from sage.groups.perm_gps.hyperoctahedral_group import _wreath_group
                sage: P = PolynomialHyperoctahedralSpecies(QQ, 2)
                sage: X = P(_wreath_group(2, 1).subgroup([]))
                sage: Y = P(_wreath_group(2, 1))
                sage: V = 2*X - 3*Y
                sage: V.is_virtual()
                True
                sage: (X*Y).is_virtual()
                False
            """
            return any(c < 0 for c in self.coefficients(sort=False))

        def is_molecular(self):
            r"""
            Return whether ``self`` is a molecular `r`-species.

            EXAMPLES::

                sage: from sage.rings.species_hyperoctahedral import PolynomialHyperoctahedralSpecies
                sage: from sage.groups.perm_gps.hyperoctahedral_group import _wreath_group
                sage: P = PolynomialHyperoctahedralSpecies(QQ, 2)
                sage: X = P(_wreath_group(2, 1).subgroup([]))
                sage: Y = P(_wreath_group(2, 1))
                sage: (2*X).is_molecular()
                False
                sage: (X*Y).is_molecular()
                True
            """
            coefficients = self.coefficients(sort=False)
            return len(coefficients) == 1 and coefficients[0] == 1

        def is_atomic(self):
            r"""
            Return whether ``self`` is an atomic `r`-species.

            EXAMPLES::

                sage: from sage.rings.species_hyperoctahedral import PolynomialHyperoctahedralSpecies
                sage: from sage.groups.perm_gps.hyperoctahedral_group import _wreath_group
                sage: P = PolynomialHyperoctahedralSpecies(QQ, 2)
                sage: X = P(_wreath_group(2, 1).subgroup([]))
                sage: Y = P(_wreath_group(2, 1))
                sage: (2*Y).is_atomic()
                False
                sage: (X*Y).is_atomic()
                False
                sage: Y.is_atomic()
                True
            """
            return self.is_molecular() and self.support()[0].is_atomic()
