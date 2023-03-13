from unittest import TestCase

from numpy import isscalar
from loguniform import LogUniform as dist


class test_constructor(TestCase):
    def test1(self):
        with self.assertRaises(TypeError):
            dist(a=1)

        with self.assertRaises(TypeError):
            dist(b=1000)
    
    def test2(self):
        with self.assertRaises(AssertionError):
            dist(a=10, b=1)

    def test3(self):
        with self.assertRaises(AssertionError):
            dist(a=0, b=1)
        
        with self.assertRaises(AssertionError):
            dist(a=0, b=0)

    def test4(self):
        d = dist(a=1, b=100)
        self.assertEqual(d.a, 1)
        self.assertEqual(d.b, 100)

        d = dist(a=10.3, b=665.1)
        self.assertEqual(d.a, 10.3)
        self.assertEqual(d.b, 665.1)


class test_methods(TestCase):
    def test_pdf(self):
        try:
            from scipy.stats import reciprocal
            from numpy.random import randint, uniform

            a = randint(1, 100)
            b = a + randint(1, 1000)
            d = dist(a, b)

            for _ in range(100):
                x = uniform(a, b)
                self.assertAlmostEqual(d.pdf(x), reciprocal(a, b).pdf(x))

        except ImportError:
            pass  # ok, no luck checking things with scipy...

        d = dist(a=10, b=5000)
        self.assertEqual(d.pdf(0), 0.0)
        self.assertEqual(d.pdf(6000), 0.0)

        self.assertNotEqual(d.pdf(d.a), 0.0)
        self.assertGreater(d.pdf(d.a), 0.0)

        self.assertNotEqual(d.pdf(d.b), 0.0)
        self.assertGreater(d.pdf(d.b), 0.0)

    def test_rvs(self):
        d = dist(a=1, b=10)
        self.assertTrue(isscalar(d.rvs()))
        self.assertGreaterEqual(d.rvs(), 1.0)
        self.assertLessEqual(d.rvs(), 10.0)

        self.assertTrue(d.rvs(25).size == 25)
        self.assertTrue((d.rvs(25) >= 1.0).all())
        self.assertTrue((d.rvs(25) <= 10.0).all())
