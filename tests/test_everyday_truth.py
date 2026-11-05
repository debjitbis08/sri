import unittest

import numpy as np

from src import everyday_truth as T


class TestTruth(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.st, cls.world = T.build()

    def test_k3(self):
        self.assertEqual(T.k3_and('T', 'U'), 'U')
        self.assertEqual(T.k3_and('F', 'U'), 'F')
        self.assertEqual(T.k3_or('T', 'U'), 'T')
        self.assertEqual(T.k3_or('F', 'U'), 'U')
        self.assertEqual(T.NOT['U'], 'U')

    def test_quantities(self):
        self.assertEqual(T.quantity('bird', 'has', 'feathers'), 'always')
        self.assertEqual(T.quantity('bird', 'can', 'fly'), 'usually')
        self.assertEqual(T.quantity('fish', 'has', 'fur'), 'never')
        self.assertEqual(T.quantity('mammal', 'can', 'swim'), 'sometimes')

    def test_counts(self):
        st = self.st
        self.assertEqual(len(st.a), 18650)
        self.assertEqual(int((st.kind == 'kind').sum()), 150)
        self.assertEqual(int((st.kind == 'connective').sum()), 21)
        self.assertEqual(int((st.test == 'S4a').sum()), 44)

    def test_unknown_only_where_open(self):
        st = self.st
        u = st.v == T.VALUES.index('U')
        ind = st.kind == 'individual'
        self.assertTrue(np.all(st.sub[u & ind] == 'unknown-colour'))
        names = np.array(st.entities)
        self.assertTrue(set(names[st.a[u & ind]]) <= set(T.colourless_animals()))

    def test_anon_defaults(self):
        st = self.st
        names = np.array(st.entities)
        i = np.flatnonzero((names[st.a] == 'bird-1') & (np.array(st.relations)[st.r] == 'can') & (names[st.b] == 'swim'))[0]
        self.assertEqual(T.VALUES[st.v[i]], 'U')   # sometimes -> unknown
        i = np.flatnonzero((names[st.a] == 'bird-1') & (np.array(st.relations)[st.r] == 'can') & (names[st.b] == 'fly'))[0]
        self.assertEqual(T.VALUES[st.v[i]], 'T')   # usually -> T


if __name__ == '__main__':
    unittest.main()
