import unittest
import numpy as np
from numpy.testing import assert_allclose
import projgrad


class TestProjection(unittest.TestCase):
    """ Test projection onto probability simplex for simple examples. """
    def setUp(self):
        self.atol = 1e-12
        self.rtol = 1e-12

    def test_inside(self):
        "if point already on simplex it should be returned unchanged"
        assert_allclose(projgrad.project(np.array([1.0, 0.0])),
                        np.array([1.0, 0.0]),
                        atol=self.atol, rtol=self.rtol)
        assert_allclose(projgrad.project(np.array([0.6, 0.4])),
                        np.array([0.6, 0.4]),
                        atol=self.atol, rtol=self.rtol)

    def test_nonnormalized(self):
        "test with non-normalized inputs"
        assert_allclose(projgrad.project(np.array([0.5, 0.0])),
                        np.array([0.75, 0.25]),
                        atol=self.atol, rtol=self.rtol)
        assert_allclose(projgrad.project(np.array([0.25, 0.25])),
                        np.array([0.5, 0.5]),
                        atol=self.atol, rtol=self.rtol)
        assert_allclose(projgrad.project(np.array([0.3, 0.5])),
                        np.array([0.4, 0.6]),
                        atol=self.atol, rtol=self.rtol)

    def test_outside(self):
        "test with inputs containing values < 0 and > 1"
        assert_allclose(projgrad.project(np.array([1.0, -1.0])),
                        np.array([1.0, 0.0]),
                        atol=self.atol, rtol=self.rtol)
        assert_allclose(projgrad.project(np.array([-1.0, 2.0])),
                        np.array([0.0, 1.0]),
                        atol=self.atol, rtol=self.rtol)
        assert_allclose(projgrad.project(np.array([1.0, 0.5, -1.0])),
                        np.array([0.75, 0.25, 0.0]),
                        atol=self.atol, rtol=self.rtol)

    def test_masked(self):
        "test with mask for entries that don't get projected"
        assert_allclose(projgrad.project(np.array([0.5, 0.0]),
                                         mask=[False, False]),
                        np.array([0.75, 0.25]),
                        atol=self.atol, rtol=self.rtol)
        assert_allclose(projgrad.project(np.array([0.7, 0.7, 0.0]),
                                         mask=[False, False, True]),
                        np.array([0.5, 0.5, 0.0]),
                        atol=self.atol, rtol=self.rtol)
        assert_allclose(projgrad.project(np.array([0.3, 0.3, 0.0]),
                                         mask=[False, False, True]),
                        np.array([0.5, 0.5, 0.0]),
                        atol=self.atol, rtol=self.rtol)


class TestMinimize(unittest.TestCase):
    """ Test projected gradient minimization algorithm on simple examples. """
    def setUp(self):
        self.atol = 1e-5
        self.rtol = 1e-5

        def objective(x):
            return np.sum(x**2), 2 * x
        self.objective = objective

    def test_simple(self):
        assert_allclose(projgrad.minimize(self.objective, [0.3, 0.7]),
                        np.array([0.5, 0.5]),
                        atol=self.atol, rtol=self.rtol)

    def test_masked(self):
        assert_allclose(projgrad.minimize(self.objective, [0.1, 0.9, 0.0],
                                          mask=[False, False, True]),
                        np.array([0.5, 0.5, 0.0]),
                        atol=self.atol, rtol=self.rtol)
        assert_allclose(projgrad.minimize(self.objective, [0.2, 0.7, 0.5],
                                          mask=[False, False, True]),
                        np.array([0.25, 0.25, 0.5]),
                        atol=self.atol, rtol=self.rtol)

if __name__ == '__main__':
    unittest.main()
