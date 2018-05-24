from unittest import TestCase

import loguniform
dist = loguniform.ModifiedLogUniform

class test_constructor(TestCase):
    def test1(self):
        with self.assertRaises(TypeError):
            dist(a=1)

        with self.assertRaises(TypeError):
            dist(b=1000)
    
        with self.assertRaises(TypeError):
            dist(knee=10)

    def test2(self):
        with self.assertRaises(AssertionError):
            dist(knee=10, b=1)

    def test3(self):
        with self.assertRaises(AssertionError):
            dist(knee=0, b=1)
        
        with self.assertRaises(AssertionError):
            dist(knee=0, b=0)

    def test4(self):
        d = dist(knee=1, b=100)
        self.assertEqual(d.knee, 1)
        self.assertEqual(d.b, 100)

        d = dist(knee=10.3, b=665.1)
        self.assertEqual(d.knee, 10.3)
        self.assertEqual(d.b, 665.1)


class test_methods(TestCase):
    def test_pdf(self):

        d = dist(knee=10, b=5000)
        self.assertEqual(d.pdf(6000), 0)
        
        self.assertNotEqual(d.pdf(d.a), 0)
        self.assertGreater(d.pdf(d.a), 0)
        
        self.assertGreater(d.pdf(d.knee), 0)
        self.assertEqual(d.pdf(-1), 0)

        self.assertNotEqual(d.pdf(d.b), 0)
        self.assertGreater(d.pdf(d.b), 0)
