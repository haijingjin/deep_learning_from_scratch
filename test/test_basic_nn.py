import unittest
from src.basic_nn import sigmoid, sigmoid_prime

class TestSigmoidFunctions(unittest.TestCase):
    def test_sigmoid(self):
        self.assertAlmostEqual(sigmoid(0), 0.5, places=5, msg="sigmoid(0) should be 0.5")
        self.assertAlmostEqual(sigmoid(1), 0.7310585786, places=5, msg="sigmoid(1) should be approximately 0.731")
    
    def test_sigmoid_prime(self):
        self.assertAlmostEqual(sigmoid_prime(0), 0.25, places=5, msg="sigmoid_prime(0) should be 0.25")
        self.assertAlmostEqual(sigmoid_prime(1), 0.1966119332, places=5, msg="sigmoid_prime(1) should be approximately 0.196")

if __name__ == '__main__':
    unittest.main()