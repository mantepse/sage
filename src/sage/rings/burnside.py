from sage.misc.cachefunc import cached_method
from sage.structure.parent import Parent
from sage.structure.element import Element
from sage.structure.element import parent
from sage.structure.unique_representation import UniqueRepresentation
from sage.rings.integer_ring import ZZ
from sage.categories.finite_enumerated_sets import FiniteEnumeratedSets
from sage.categories.sets_with_grading import SetsWithGrading
from sage.categories.graded_algebras_with_basis import GradedAlgebrasWithBasis
from sage.categories.algebras import Algebras
from sage.categories.monoids import Monoids
from sage.groups.perm_gps.permgroup import PermutationGroup, PermutationGroup_generic
from sage.groups.perm_gps.permgroup_named import SymmetricGroup
from sage.libs.gap.libgap import libgap
from sage.combinat.free_module import CombinatorialFreeModule
from sage.monoids.indexed_free_monoid import IndexedFreeAbelianMonoid, IndexedMonoid

GAP_FAIL = libgap.eval('fail')

def _is_conjugate(G, H1, H2):
    r"""
    Test if ``H1`` and ``H2`` are conjugate subgroups in ``G``.

    EXAMPLES::

        sage: G = SymmetricGroup(3)
        sage: H1 = PermutationGroup([(1,2)])
        sage: H2 = PermutationGroup([(2,3)])
        sage: from sage.rings.burnside import _is_conjugate
        sage: _is_conjugate(G, H1, H2)
        True
    """
    return GAP_FAIL != libgap.RepresentativeAction(G, H1, H2)

class ElementCache():
    def __init__(self):
        r"""
        Class for caching elements of conjugacy classes of subgroups.
        """
        self._cache = dict()

    def _group_invariants(self, H):
        r"""
        Return tuple of group invariants associated with H.
        """
        return tuple([H.order(), H.degree()])
    
    def _cache_get(self, H):
        r"""
        Return the cached element for H, or create it
        if it doesn't exist.
        """
        element = self.element_class(self, H)
        key = self._group_invariants(H)
        if key in self._cache:
            for H0, H0_elm in self._cache[key]:
                if _is_conjugate(element.subgroup_of(), H0, H):
                    return H0_elm
            else:
                self._cache[key].append((H, element))
        else:
            self._cache[key] = [(H, element)]
        return element

class ConjugacyClassOfSubgroups(Element):
    def __init__(self, parent, C):
        r"""
        Initialize the conjugacy class of ``C``.

        TESTS::

            sage: G = SymmetricGroup(4)
            sage: B = BurnsideRing(G)
            sage: Z4 = CyclicPermutationGroup(4)
            sage: TestSuite(B(Z4)).run()
        """
        Element.__init__(self, parent)
        self._C = C

    def subgroup_of(self):
        r"""
        Return the group which this conjugacy class of subgroups
        belongs to.

        TESTS::

            sage: G = SymmetricGroup(3)
            sage: from sage.rings.burnside import ConjugacyClassesOfSubgroups
            sage: C = ConjugacyClassesOfSubgroups(G)
            sage: Z3 = CyclicPermutationGroup(3)
            sage: CZ3 = C(Z3)
            sage: CZ3.subgroup_of()
            Symmetric group of order 3! as a permutation group
        """
        return self.parent()._G

    def __hash__(self):
        r"""
        Return the hash of the representative of the conjugacy class.

        TESTS::

            sage: G = SymmetricGroup(3)
            sage: B = BurnsideRing(G)
            sage: H1 = B(PermutationGroup([(1,2)]))
            sage: H2 = B(PermutationGroup([(2,3)]))
            sage: hash(H1) == hash(H2)
            True
        """
        return hash(self._C)
    
    def _repr_(self):
        r"""
        Return a string representation of ``self``.

        TESTS::

            sage: G = SymmetricGroup(3)
            sage: from sage.rings.burnside import ConjugacyClassesOfSubgroups
            sage: C = ConjugacyClassesOfSubgroups(G)
            sage: Z3 = CyclicPermutationGroup(3)
            sage: CZ3 = C(Z3)
            sage: CZ3
            ((1,2,3),)
        """
        return repr(self._C.gens_small())

    def __le__(self, other):
        r"""
        Return if this element is less than or equal to ``other``.

        ``self`` is less or equal to ``other`` if it is conjugate to
        a subgroup of ``other`` in the parent group.

        EXAMPLES:

        We recreate the poset of conjugacy classes of subgroups of
        `S_4`::

            sage: G = SymmetricGroup(4)
            sage: B = BurnsideRing(G)
            sage: P = Poset([B.basis(), lambda b, c: b <= c])
            sage: len(P.cover_relations())
            17
        """
        return (isinstance(other, ConjugacyClassOfSubgroups)
                and (GAP_FAIL != libgap.ContainedConjugates(self.subgroup_of(),
                                                            other._C,
                                                            self._C,
                                                            True)))

    def __eq__(self, other):
        r"""
        Return if this element is equal to ``other``.

        Two elements compare equal if they are conjugate subgroups in the parent group.

        TESTS::

            sage: G = SymmetricGroup(4)
            sage: B = BurnsideRing(G)
            sage: H1 = PermutationGroup([(1,2)])
            sage: H2 = PermutationGroup([(2,3)])
            sage: B[H1] == B[H2]
            True
        """
        return (isinstance(other, ConjugacyClassOfSubgroups)
                and _is_conjugate(self.subgroup_of(), self._C, other._C))

    def __lt__(self, other):
        r"""
        Return if this element is less than ``other``.

        TESTS::

            sage: G = SymmetricGroup(4)
            sage: B = BurnsideRing(G)
            sage: H1 = PermutationGroup([(1,2)])
            sage: H2 = PermutationGroup([(2,3)])
            sage: H3 = PermutationGroup([(1,2),(3,4)])
            sage: B[H1] < B[H2]
            False
            sage: B[H1] == B[H2]
            True
            sage: B[H1] < B[H3]
            True
            sage: B[H2] < B[H3]
            True
        """
        return self <= other and self != other

class ConjugacyClassesOfSubgroups(Parent, ElementCache):
    def __init__(self, G):
        r"""
        Initialize the set of conjugacy classes of ``G``.

        INPUT:

        ``G`` -- a group.

        TESTS::

            sage: from sage.rings.burnside import ConjugacyClassesOfSubgroups
            sage: G = CyclicPermutationGroup(4)
            sage: C = ConjugacyClassesOfSubgroups(G)
            sage: TestSuite(C).run()
        """
        self._G = G
        Parent.__init__(self, category=FiniteEnumeratedSets())
        ElementCache.__init__(self)

    def __eq__(self, other):
        r"""
        Return if ``self`` is equal to ``other``.

        TESTS::

            sage: from sage.rings.burnside import ConjugacyClassesOfSubgroups
            sage: Z4 = CyclicPermutationGroup(4)
            sage: D4 = DiCyclicGroup(4)
            sage: C1 = ConjugacyClassesOfSubgroups(Z4)
            sage: C2 = ConjugacyClassesOfSubgroups(D4)
            sage: C1 == C2
            False
        """
        if not isinstance(other, ConjugacyClassesOfSubgroups):
            return False
        return self._G == other._G

    def __hash__(self):
        r"""
        Return the hash of the underlying group.

        TESTS::

            sage: from sage.rings.burnside import ConjugacyClassesOfSubgroups
            sage: Z4 = CyclicPermutationGroup(4)
            sage: D4 = DiCyclicGroup(4)
            sage: C1 = ConjugacyClassesOfSubgroups(Z4)
            sage: C2 = ConjugacyClassesOfSubgroups(D4)
            sage: hash(C1) == hash(C2)
            False
        """
        return hash(self._G)

    def _element_constructor_(self, x):
        r"""
        Construct the conjugacy class of subgroups containing ``x``.

        TESTS::

            sage: from sage.rings.burnside import ConjugacyClassesOfSubgroups
            sage: G = SymmetricGroup(4)
            sage: C = ConjugacyClassesOfSubgroups(G)
            sage: Z4 = CyclicPermutationGroup(4)
            sage: C(Z4)
            ((1,2,3,4),)
        """
        if parent(x) == self:
            return x
        if isinstance(x, PermutationGroup_generic):
            if x.is_subgroup(self._G):
                return self._cache_get(x)
            raise ValueError(f"unable to convert {x} into {self}: not a subgroup of {self._G}")
        raise ValueError("unable to convert {x} into {self}")


    def __iter__(self):
        r"""
        Return iterator over conjugacy classes of subgroups of the group.

        TESTS::

            sage: G = SymmetricGroup(3)
            sage: B = BurnsideRing(G)
            sage: [g for g in B._indices]
            [((),), ((1,2),), ((1,2,3),), ((1,2,3), (1,2))]
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
        if parent(H) == self:
            return True
        return (isinstance(H, PermutationGroup_generic)
                and H.is_subgroup(self._G))

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
        The Burnside ring of the group ``G``.

        INPUT:

        ``G`` -- a group.
        ``base_ring`` -- the ring of coefficients. Default value is ``ZZ``.

        EXAMPLES::

            sage: G = SymmetricGroup(6)
            sage: B = BurnsideRing(G)
            sage: X = Subsets(6, 2)
            sage: b = B.construct_from_action(lambda g, x: X([g(e) for e in x]), X)
            sage: B.basis().keys()._cache
            {48: [((1,2)(4,5,6), (3,5,4,6))], 720: [((1,2,3,4,5,6), (1,2))]}
            sage: b^2
            B[((4,5), (4,6))] + B[((5,6), (3,4), (1,2))] + B[((1,2)(4,5,6), (3,5,4,6))]
            sage: B.basis().keys()._cache
            {6: [((4,5), (4,6))],
            8: [((5,6), (3,4), (1,2))],
            48: [((1,2)(4,5,6), (3,5,4,6))],
            720: [((1,2,3,4,5,6), (1,2))]}

            sage: G = SymmetricGroup(4)
            sage: B = BurnsideRing(G)
            sage: X = Subsets(4, 2)
            sage: b = B.construct_from_action(lambda g, x: X([g(e) for e in x]), X)
            sage: b.tensor(b)
            B[((3,4), (1,2))] # B[((3,4), (1,2))]
            sage: (b.tensor(b))^2
            B[((),)] # B[((),)] + 2*B[((),)] # B[((3,4), (1,2))] + 2*B[((3,4), (1,2))] # B[((),)] + 4*B[((3,4), (1,2))] # B[((3,4), (1,2))]

        TESTS::

            sage: G = SymmetricGroup(4)
            sage: B = BurnsideRing(G)
            sage: TestSuite(B).run()
        """
        self._G = G
        basis_keys = ConjugacyClassesOfSubgroups(G)
        category = Algebras(base_ring).Commutative().WithBasis()
        CombinatorialFreeModule.__init__(self, base_ring, basis_keys,
                                        category=category, prefix="B")
        self._indices(G).rename("1")

    def __getitem__(self, H):
        r"""
        Return the basis element indexed by ``H``.

        ``H`` must be a subgroup of the group.

        EXAMPLES::

            sage: G = SymmetricGroup(4)
            sage: B = BurnsideRing(G)
            sage: Z4 = CyclicPermutationGroup(4)
            sage: B[Z4]
            B[((1,2,3,4),)]
        """
        return self._from_dict({self._indices(H): 1})

    def construct_from_action(self, action, domain):
        r"""
        Construct an element of the Burnside ring from a group action.

        INPUT:

        - ``action`` - an action on ``domain``
        - ``domain`` - a finite set

        EXAMPLES::

            sage: G = SymmetricGroup(4)
            sage: B = BurnsideRing(G)

        We create a group action of `S_4` on two-element subsets::

            sage: X = Subsets(4, 2)
            sage: a = lambda g, x: X([g(e) for e in x])
            sage: B.construct_from_action(a, X)
            B[((3,4), (1,2))]

        Next, we create a group action of `S_4` on itself via conjugation::

            sage: X = G
            sage: a = lambda g, x: g*x*g.inverse()
            sage: B.construct_from_action(a, X)
            B[((3,4), (1,3)(2,4), (1,4)(2,3))] + B[((2,4,3),)] + B[((3,4), (1,2))] + B[((1,2,3,4),)] + 1

        TESTS::

            sage: G = SymmetricGroup(4)
            sage: B = BurnsideRing(G)
            sage: [b.support()[0]._C.order() for b in B.gens()]
            [1, 2, 2, 3, 4, 4, 4, 6, 8, 12, 24]
            sage: sorted((o, len(l)) for o, l in B._indices._cache.items())
            [(1, 1), (2, 2), (3, 1), (4, 3), (6, 1), (8, 1), (12, 1), (24, 1)]

            sage: G = SymmetricGroup(4)
            sage: B = BurnsideRing(G)
            sage: B(-3)
            -3
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
            ((1,2,3,4,5,6,7,8)(9,10,11,12,13,14,15,16), (1,9,5,13)(2,16,6,12)(3,15,7,11)(4,14,8,10))
        """
        return self._indices(self._G)

    def product_on_basis(self, H, K):
        r"""
        Return the product of the basis elements indexed by ``H`` and ``K``.

        For the symmetric group, this is also known as the Hadamard
        or tensor product of group actions.

        EXAMPLES::

            sage: G = SymmetricGroup(3)
            sage: B = BurnsideRing(G)
            sage: matrix([[b * c for b in B.gens()] for c in B.gens()])
            [            6*B[((),)]             3*B[((),)]             2*B[((),)]               B[((),)]]
            [            3*B[((),)] B[((),)] + B[((1,2),)]               B[((),)]            B[((1,2),)]]
            [            2*B[((),)]               B[((),)]        2*B[((1,2,3),)]          B[((1,2,3),)]]
            [              B[((),)]            B[((1,2),)]          B[((1,2,3),)]                      1]

        TESTS::

            sage: G = SymmetricGroup(3)
            sage: B = BurnsideRing(G)
            sage: Z3 = CyclicPermutationGroup(3)
            sage: Z2 = CyclicPermutationGroup(2)
            sage: from sage.rings.burnside import ConjugacyClassesOfSubgroups
            sage: C = ConjugacyClassesOfSubgroups(G)
            sage: B.product_on_basis(C(Z2), C(Z3))
            B[((),)]
        """
        g_reps = [rep for rep, size in libgap.DoubleCosetRepsAndSizes(self._G, H._C, K._C)]
        from collections import Counter
        C = Counter()
        for g in g_reps:
            g_sup_K = libgap.ConjugateSubgroup(K._C, g)
            P = self._G.subgroup(gap_group=libgap.Intersection(H._C, g_sup_K))
            C[self._indices(P)] += 1
        return self._from_dict(dict(C))

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

class AtomicConjugacyClass(ConjugacyClassOfSubgroups):
    def __init__(self, parent, C):
        r"""
        An atomic conjugacy class.
        """
        ConjugacyClassOfSubgroups.__init__(self, parent, C)

    @cached_method
    def subgroup_of(self):
        r"""
        Return the group which this atomic conjugacy class of subgroups
        belongs to.

        TESTS::

            sage: At = UnivariateAtomicConjugacyClasses()
            sage: g = At(CyclicPermutationGroup(3)); g
            {3, [(1,2,3)]}
            sage: g.subgroup_of()
            Symmetric group of order 3! as a permutation group
        """
        return SymmetricGroup(self._C.degree())
    
    def _repr_(self):
        r"""
        Return a string representation of ``self``.

        EXAMPLES::

            sage: At = UnivariateAtomicConjugacyClasses()
            sage: g = At(CyclicPermutationGroup(3)); g
            {3, [(1,2,3)]}
        """
        return "{" + f"{self._C.degree()}, {self._C.gens_small()}" + "}"

class UnivariateAtomicConjugacyClasses(UniqueRepresentation, Parent, ElementCache):
    def __init__(self):
        r"""
        Infinite set of atomic conjugacy classes of `S_n`.

        Graded by `n \geq 0`.
        """
        category = SetsWithGrading().Infinite()
        Parent.__init__(self, category=category)
        ElementCache.__init__(self)
    
    @cached_method
    def an_element(self):
        return self._cache_get(SymmetricGroup(0))

    def _element_constructor_(self, x):
        r"""
        Construct the atomic conjugacy class of subgroups containing ``x``.
        """
        # BUG: SymmetricGroup(0) fails to be constructed.
        if parent(x) == self:
            return x
        if isinstance(x, PermutationGroup_generic):
            if len(x.disjoint_direct_product_decomposition()) == 1:
                return self._cache_get(x)
            else:
                raise ValueError(f"{x} is not atomic")
        raise ValueError(f"unable to convert {x} into {self}")
    
    def __contains__(self, H):
        r"""
        Return if H is an atomic subgroup.
        """
        if parent(H) == self:
            return True
        return (isinstance(H, PermutationGroup_generic)
                and len(H.disjoint_direct_product_decomposition()) == 1)
    
    def _repr_(self):
        r"""
        Return a string representation of ``self``.

        EXAMPLES::

            sage: At = UnivariateAtomicConjugacyClasses()
            sage: At
            Set of all atomic conjugacy classes on 1 sort
        """
        return "Set of all atomic conjugacy classes on 1 sort"
    
    Element=AtomicConjugacyClass

class UnivariateMolecularConjugacyClasses(IndexedFreeAbelianMonoid):
    @staticmethod
    def __classcall__(cls):
        r"""
        Needed to make this work. Taken from example in center_uea.py.
        """
        return super(IndexedMonoid, cls).__classcall__(cls)

    def __init__(self):
        r"""
        Infinite set of molecular conjugacy classes of `S_n`.
        Implemented as elements of the free abelian monoid on
        univariate atomic conjugacy classes.

        Graded by `n \geq 0`.
        """
        category = Monoids() & SetsWithGrading().Infinite()
        IndexedFreeAbelianMonoid.__init__(self,
                                          indices=UnivariateAtomicConjugacyClasses(),
                                          prefix='', bracket=False,
                                          category=category)

    def _project(self, H, part):
        r"""
        Project `H` onto a subset ``part`` of its domain.
        ``part`` must be a union of cycles, but this is not checked.
        """
        restricted_gens = [[cyc for cyc in gen.cycle_tuples() if cyc[0] in part] for gen in H.gens_small()]
        mapping = {elm: i for i, elm in enumerate(part, 1)}
        normalized_gens = [[tuple(mapping[x] for x in cyc) for cyc in gen] for gen in restricted_gens]
        return PermutationGroup(gens=normalized_gens)
    
    def _element_constructor_(self, x=None):
        if x is None:
            return super()._element_constructor_(x)
        if parent(x) == self:
            return x
        if isinstance(x, PermutationGroup_generic):
            domain_partition = x.disjoint_direct_product_decomposition()
            return super()._element_constructor_([(self.gen(self._project(x, part)), 1) for part in domain_partition])
        raise ValueError("unable to convert {x} into {self}")

    def _repr_(self):
        r"""
        Return a string representation of ``self``.

        EXAMPLES::

            sage: Mol = UnivariateMolecularConjugacyClasses()
            sage: Mol
            Set of all molecular conjugacy classes on 1 sort
        """
        return "Set of all molecular conjugacy classes on 1 sort"

class PolynomialMolecularDecomposition(CombinatorialFreeModule):
    def __init__(self, base_ring=ZZ):
        basis_keys = UnivariateMolecularConjugacyClasses()
        category = GradedAlgebrasWithBasis(base_ring).Commutative()
        CombinatorialFreeModule.__init__(self, base_ring,
                                        basis_keys=basis_keys,
                                        category=category,
                                        prefix='', bracket=False)

    def __getitem__(self, H):
        r"""
        Return the basis element indexed by ``H``.

        ``H`` must be a subgroup of a symmetric group.

        EXAMPLES::

            sage: P = PolynomialMolecularDecomposition()
            sage: Z4 = CyclicPermutationGroup(4)
            sage: P[Z4]
            {4, [(1,2,3,4)]}
        """
        return self._from_dict({self._indices(H): 1})

    @cached_method
    def one_basis(self):
        r"""
        Returns SymmetricGroup(0), which indexes the one of this algebra,
        as per :meth:`AlgebrasWithBasis.ParentMethods.one_basis`.

        EXAMPLES::

            sage: P = PolynomialMolecularDecomposition()
            sage: P.one_basis()
            1
        """
        return self._indices(SymmetricGroup(0))

    def product_on_basis(self, H, K):
        r"""
        Return the product of the basis elements indexed by `H` and `K`.

        Let `H` be a subgroup of `\mathfrak{G}_n` and `K` be a subgroup of `\mathfrak{G}_m`.
        Then we define their Cauchy product as the subgroup of `\mathfrak{G}_{n+m}` given by
        `H \ast K = \{h \dot k \vert h \in H_{\{1 \ldots n\}}, K_{\{n+1 \ldots n+m\}}\}` where
        the subscripts denote the domains on which H and K act. Note that this is isomorphic to
        the direct product of `H` and `K`.

        EXAMPLES::

            sage: Mol = UnivariateMolecularConjugacyClasses()
            sage: P = PolynomialMolecularDecomposition()
            sage: L1 = SymmetricGroup(3).conjugacy_classes_subgroups()
            sage: L2 = SymmetricGroup(2).conjugacy_classes_subgroups()
            sage: matrix([[P.product_on_basis(Mol(x),Mol(y)) for x in L1] for y in L2])
            [                       {1, [()]}^5           {1, [()]}^3*{2, [(1,2)]}         {3, [(1,2,3)]}*{1, [()]}^2  {3, [(1,2,3), (2,3)]}*{1, [()]}^2]
            [          {1, [()]}^3*{2, [(1,2)]}           {1, [()]}*{2, [(1,2)]}^2        {3, [(1,2,3)]}*{2, [(1,2)]} {3, [(1,2,3), (2,3)]}*{2, [(1,2)]}]
        """        
        return self._from_dict({H * K: 1})

    def _repr_(self):
        r"""
        Return a string representation of ``self``.

        EXAMPLES::

            sage: P = PolynomialMolecularDecomposition()
            sage: P
            Polynomial Molecular Decomposition
        """
        return "Polynomial Molecular Decomposition"
