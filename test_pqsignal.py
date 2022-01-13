import unittest
from pqsignal import generate
import numpy as np
import matplotlib.pyplot as plt

class TestPQSignalCases(unittest.TestCase):

    def setUp(self):
        self.sps = 2000
        self.t = np.arange(-0.5,0.5,1/self.sps)

    def test_case1(self):
        signal = generate(1, self.t, self.sps)
        plt.plot(self.t, signal)
        plt.title('T1')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude (pu)')
        plt.show()

    def test_case2(self):
        signal = generate(2, self.t, self.sps)
        plt.plot(self.t, signal)
        plt.title('T2')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude (pu)')
        plt.show()

    def test_case3(self):
        signal = generate(3, self.t, self.sps)
        plt.plot(self.t, signal)
        plt.title('T3')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude (pu)')
        plt.show()

    def test_case4(self):
        signal = generate(4, self.t, self.sps)
        plt.plot(self.t, signal)
        plt.title('T4')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude (pu)')
        plt.show()

    def test_case5(self):
        signal = generate(5, self.t, self.sps)
        plt.plot(self.t, signal)
        plt.title('T5')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude (pu)')
        plt.show()

    def test_case6(self):
        signal = generate(6, self.t, self.sps)
        plt.plot(self.t, signal)
        plt.title('T6')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude (pu)')
        plt.show()

    def test_case7(self):
        signal = generate(7, self.t, self.sps)
        plt.plot(self.t, signal)
        plt.title('T7')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude (pu)')
        plt.show()

    def test_case8(self):
        signal = generate(8, self.t, self.sps)
        plt.plot(self.t, signal)
        plt.title('T8')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude (pu)')
        plt.show()


if __name__ == '__main__':
    unittest.main()