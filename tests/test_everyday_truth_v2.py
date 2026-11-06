import unittest

import numpy as np

from src import everyday_truth_v2 as T2


class TestTruthV2(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.st, _ = T2.build()

    def q(self, rel, a, b):
        st = self.st
        names, rels = np.array(st.entities), np.array(st.relations)
        return np.flatnonzero((rels[st.r] == rel) & (names[st.a] == a) & (names[st.b] == b))[0]

    def test_kind_chain(self):
        self.assertEqual(T2.kind_chain('robin'), {'bird': 1, 'vertebrate': 2, 'animal': 3, 'living-thing': 4})
        self.assertEqual(T2.kind_chain('ant')['invertebrate'], 2)

    def test_closures(self):
        st = self.st
        i = self.q('ancestor-of', 'linea1', 'linea5')
        self.assertEqual((st.world_y[i], st.test[i], st.sub[i]), (1, 'L1-anc', '4'))
        i = self.q('ancestor-of', 'linea5', 'linea1')
        self.assertEqual((st.world_y[i], st.test[i]), (0, 'L1-anc'))
        i = self.q('ancestor-of', 'linea1', 'linea3')
        self.assertEqual((st.world_y[i], st.test[i]), (1, ''))
        i = self.q('bigger-than', 'whale', 'lion')
        self.assertEqual((st.world_y[i], st.test[i], st.sub[i]), (1, 'L1-size', '3'))
        i = self.q('kind-of', 'robin', 'living-thing')
        self.assertEqual((st.world_y[i], st.test[i]), (1, 'L1-kind'))
        i = self.q('kind-of', 'robin', 'vertebrate')
        self.assertEqual((st.world_y[i], st.test[i]), (1, ''))

    def test_named(self):
        st = self.st
        i = self.q('named-can', 'eagle-w', 'fly')
        self.assertEqual((st.world_y[i], st.test[i]), (1, 'L2-named'))
        i = self.q('named-can', 'duck-w', 'swim')
        self.assertEqual((st.world_y[i], st.test[i]), (1, ''))

    def test_size(self):
        self.assertEqual(len(self.st.a), 22865)


if __name__ == '__main__':
    unittest.main()
