from sage.misc.cachefunc import cached_method
from sage.structure.unique_representation import UniqueRepresentation
from sage.structure.parent import Parent
from sage.structure.element import Element
from sage.rings.integer_ring import ZZ
from sage.structure.formal_sum import FormalSum
from sage.categories.sets_cat import cartesian_product
from sage.categories.finite_enumerated_sets import FiniteEnumeratedSets
from sage.groups.perm_gps.permgroup import PermutationGroup
from sage.libs.gap.libgap import libgap
from sage.categories.algebras import Algebras
from sage.combinat.free_module import CombinatorialFreeModule

def _is_conjugate(G, H1, H2):
    r"""
    Test if ``H1`` and ``H2`` are conjugate subgroups in ``G``.

    EXAMPLES:

        sage: G = SymmetricGroup(3)
        sage: H1 = PermutationGroup([(1,2)])
        sage: H2 = PermutationGroup([(2,3)])
        sage: from sage.rings.burnside import _is_conjugate
        sage: _is_conjugate(G, H1, H2)
        True
    """
    return libgap.eval('fail') != libgap.RepresentativeAction(G, H1, H2)

class ConjugacyClassOfSubgroups(Element):
    def __init__(self, parent, C):
        Element.__init__(self, parent)
        self._C = C

    def __hash__(self):
        r"""
        Return the hash of the representative of the conjugacy class.
        """
        return hash(self._C)

    def _repr_(self):
        r"""
        Return a string representation of ``self``.
        """
        name = self.parent()._names.get(self._C, None)
        return repr(self._C.gens()) if name is None else name

    def __le__(self, other):
        r"""
        Return if this element is less or equal to ``other``.

        ``self`` is less or equal to ``other`` if it is conjugate to
        a subgroup of ``other`` in the parent group.
        """
        return libgap.eval('fail') != libgap.ContainedConjugates(self.parent()._G, other._C, self._C, True)

    def __eq__(self, other):
        r"""
        Return if this element is equal to ``other``.

        Two elements compare equal if they are conjugate subgroups in the parent group.
        """
        return _is_conjugate(self.parent()._G, self._C, other._C)

class ConjugacyClassesOfSubgroups(Parent):
    def __init__(self, G):
        r"""
        INPUT:

        ``G`` -- a group.
        """
        self._G = G
        self._cache = dict() # invariant to subgroups
        self._names = dict() # dictionary mapping subgroups to names
        Parent.__init__(self, category=FiniteEnumeratedSets())

    def _group_invariant(self, H):
        r"""
        Return the set of group invariants associated with ``H``.
        """
        return H.order()

    def _normalize(self, H):
        r"""
        Add ``H`` as the representative of its conjugacy class to the cache and return it.

        If a representative of the conjugacy class of ``H``
        already exists in the cache, return that.
        """
        if not H.is_subgroup(self._G):
            raise ValueError(f"{H} is not a subgroup of {self._G}")
        p = self._group_invariant(H)
        if p in self._cache:
            for H0 in self._cache[p]:
                if _is_conjugate(self._G, H, H0):
                    return H0
            else:
                self._cache[p].append(H)
        else:
            self._cache[p] = [H]
        return H

    def _element_constructor_(self, x):
        r"""
        Construct the conjugacy class of subgroups containing ``x``.
        """
        if x.is_subgroup(self._G):
            return self.element_class(self, self._normalize(x))
        raise ValueError(f"unable to convert {x} into {self}: not a subgroup of " + repr(self._G))

    def set_name(self, H, name):
        r"""
        Rename conjugacy class of ``H`` to ``name``.

        Passing ``None`` to ``name`` will remove any previously assigned name.
        """
        if not isinstance(name, str):
            raise TypeError("name must be a string")
        H_norm = self._normalize(H)
        self._names[H_norm] = name

    def __iter__(self):
        r"""
        Return iterator over conjugacy classes of subgroups of the group.

        TESTS::

            sage: G = SymmetricGroup(3)
            sage: B = BurnsideRing(G)
            sage: [g for g in B._indices]
            [((),), ((2,3),), ((1,2,3),), 1]
        """
        return iter(self(H) for H in self._G.conjugacy_classes_subgroups())

    def __contains__(self, H):
        r"""
        Return if ``H`` is a subgroup of the group.

        TESTS::

            sage: G = SymmetricGroup(4)
            sage: B = BurnsideRing(G)
            sage: Z4 = CyclicPermutationGroup(4)
            sage: Z4 in B._indices
            True
        """
        return H.is_subgroup(self._G)

    def _repr_(self):
        r"""
        Return a string representation of ``self``.

        TESTS::

            sage: G = SymmetricGroup(4)
            sage: B = BurnsideRing(G)
            sage: B._indices
            Conjugacy classes of subgroups of Symmetric group of order 4! as a permutation group
        """
        return "Conjugacy classes of subgroups of " + repr(self._G)

    Element = ConjugacyClassOfSubgroups

class BurnsideRing(CombinatorialFreeModule):
    def __init__(self, G, base_ring=ZZ):
        r"""
        INPUT:

        ``G`` -- a group.
        ``base_ring`` -- the ring of coefficients. Default value is ``ZZ``.

        TESTS::

            sage: G = SymmetricGroup(4)
            sage: B = BurnsideRing(G)
            sage: TestSuite(B).run()
        """
        self._G = G
        basis = ConjugacyClassesOfSubgroups(G)
        basis.set_name(G, "1")
        category = Algebras(base_ring).Commutative().WithBasis()
        CombinatorialFreeModule.__init__(self, base_ring, basis, category=category)

    def __getitem__(self, H):
        r"""
        Return the basis element indexed by ``H``.

        ``H`` must be a subgroup of the group.

        EXAMPLES::

            sage: G = SymmetricGroup(4)
            sage: B = BurnsideRing(G)
            sage: Z4 = CyclicPermutationGroup(4)
            sage: B[Z4]
            B[((1,3,2,4), (1,2)(3,4))]
        """
        return self._from_dict({self._indices(H): 1})

    def construct_from_action(self, action, domain):
        r"""
        Construct an element of the Burnside ring from a group action.

        INPUT:

        - ``action`` - an action on ``domain``
        - ``domain`` - a finite set

        EXAMPLES:

            sage: G = SymmetricGroup(4)
            sage: B = BurnsideRing(G)

        We create a group action of `S_4` on two-element subsets::

            sage: X = Subsets(4, 2)
            sage: a = lambda g, x: X([g(e) for e in x])
            sage: B.construct_from_action(a, X)
            B[((3,4), (1,2)(3,4))]

        Next, we create a group action of `S_4` on itself via conjugation::

            sage: X = G
            sage: a = lambda g, x: g*x*g.inverse()
            sage: B.construct_from_action(a, X)
            B[1] + B[((1,4)(2,3), (1,3)(2,4), (3,4))] + B[((2,4,3),)] + B[((3,4), (1,2)(3,4))] + B[((1,3,2,4), (1,2)(3,4))]

        TESTS::

            sage: G = SymmetricGroup(4)
            sage: B = BurnsideRing(G)
            sage: [list(b.monomial_coefficients().keys())[0]._C.order() for b in B.gens()]
            [1, 2, 2, 3, 4, 4, 4, 6, 8, 12, 24]
            sage: sorted((o, len(l)) for o, l in B._indices._cache.items())
            [(1, 1), (2, 2), (3, 1), (4, 3), (6, 1), (8, 1), (12, 1), (24, 1)]

            sage: G = SymmetricGroup(4)
            sage: B = BurnsideRing(G)
            sage: B(-3)
            -3*B[1]
        """
        def find_stabilizer(action, pnt):
            stabilizer = []
            for g in self._G:
                if action(g, pnt) == pnt:
                    stabilizer.append(g)
            H = self._G.subgroup(stabilizer)
            gens = H.gens_small()
            return self._G.subgroup(gens)

        H = PermutationGroup(self._G.gens(), action=action, domain=domain)
        # decompose H into orbits
        orbit_list = H.orbits()
        # find the stabilizer subgroups
        stabilizer_list = [find_stabilizer(action, orbit[0]) for orbit in orbit_list]
        # normalize each summand and collect terms
        from collections import Counter
        C = Counter([self._indices(stabilizer) for stabilizer in stabilizer_list])
        return self._from_dict(dict(C))

    @cached_method
    def one_basis(self):
        r"""
        Returns the underlying group, which indexes the one of this algebra,
        as per :meth:`AlgebrasWithBasis.ParentMethods.one_basis`.

        EXAMPLES::

            sage: G = DiCyclicGroup(4)
            sage: B = BurnsideRing(G)
            sage: B.one_basis()
            1
        """
        return self._indices(self._G)

    def product_on_basis(self, g1, g2):
        r"""
        Return the product of the basis elements indexed by ``g1`` and ``g2``.

        For the symmetric group, this is also known as the Hadamard
        or tensor product of group actions.

        EXAMPLES::

            sage: G = SymmetricGroup(3)
            sage: B = BurnsideRing(G)
            sage: matrix([[b * c for b in B.gens()] for c in B.gens()])
            [            6*B[((),)]             3*B[((),)]             2*B[((),)]               B[((),)]]
            [            3*B[((),)] B[((2,3),)] + B[((),)]               B[((),)]            B[((2,3),)]]
            [            2*B[((),)]               B[((),)]        2*B[((1,2,3),)]          B[((1,2,3),)]]
            [              B[((),)]            B[((2,3),)]          B[((1,2,3),)]                   B[1]]
        """
        #TODO: Find faster way to multiply
        assert g1.parent() == g2.parent()
        G = g1.parent()._G
        dom1 = [frozenset(g) for g in G.cosets(g1._C, side="left")]
        dom2 = [frozenset(g) for g in G.cosets(g2._C, side="left")]
        domain = cartesian_product([dom1, dom2])

        def action(g, pair):
            return (frozenset(g * h for h in pair[0]),
                    frozenset(g * h for h in pair[1]))

        return self.construct_from_action(action, domain)

    def group(self):
        r"""
        Return the underlying group.

        EXAMPLES::

            sage: G = DiCyclicGroup(4)
            sage: B = BurnsideRing(G)
            sage: B.group()
            Dicyclic group of order 16 as a permutation group
        """
        return self._G

    def _repr_(self):
        r"""
        Return a string representation of ``self``.

        EXAMPLES::

            sage: G = SymmetricGroup(4)
            sage: B = BurnsideRing(G)
            sage: B
            Burnside ring of Symmetric group of order 4! as a permutation group
        """
        return "Burnside ring of " + repr(self._G)