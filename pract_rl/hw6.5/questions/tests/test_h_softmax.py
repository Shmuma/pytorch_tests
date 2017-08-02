import numpy as np
import unittest

from questions import model

import torch
from torch.autograd import Variable


class TestHSoftmax(unittest.TestCase):
    def test_softmax(self):
        # one batch, top note has high prob of left node, second level has right and left probs
        scores = Variable(torch.from_numpy(np.array([[10.0, -2.0, 3.0]], dtype=np.float32)))
        # this sample corresponds to high-prob branch, so loss should be low
        class_indices = [int("01", base=2)]
        hsm = model.HierarchicalSoftmaxMappingModule()
        res = hsm(scores, class_indices)
        self.assertAlmostEqual(res.data.cpu().numpy()[0], 0.1270, places=3)

        # same scores, but sample correspond to low-prob path to tree, loss should be high
        class_indices = [int("11", base=2)]
        hsm = model.HierarchicalSoftmaxMappingModule()
        res = hsm(scores, class_indices)
        self.assertAlmostEqual(res.data.cpu().numpy()[0], 13.0482, places=3)

    # def test_gather(self):
    #     t = torch.arange(0, 12).view(4, 3)
    #     idx = torch.LongTensor([[0, 0, 0], [3, 3, 3]])
    #     r = torch.gather(t, dim=0, index=idx)
    #     print(t)
    #
    # def test_prepend(self):
    #     t = torch.arange(0, 12).view(4, 3)
    #     z = torch.zeros(4, 1)
    #     r = torch.cat([z, t], dim=1)
    #     print(r)


class TestTwoLevelSoftmax(unittest.TestCase):
    def test_softmax(self):
        hsm = model.TwoLevelSoftmaxMappingModule(3, 4, freq_ratio=0.9999, count_of_classes=1)
        scores = Variable(torch.from_numpy(np.array([[10.0, -2.0, 3.0]], dtype=np.float32)))



if __name__ == '__main__':
    unittest.main()
