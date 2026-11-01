import unittest

import torch

from stochopt.optimizers import MomentumSGD


class MomentumSGDTests(unittest.TestCase):
    def test_matches_pytorch_sgd_across_updates(self):
        for momentum, decay in [(0.0, 0.0), (0.9, 0.01), (0.5, 0.0)]:
            with self.subTest(momentum=momentum, decay=decay):
                custom_parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
                torch_parameter = torch.nn.Parameter(custom_parameter.detach().clone())
                custom = MomentumSGD([custom_parameter], lr=0.1, momentum=momentum,
                                     weight_decay=decay)
                reference = torch.optim.SGD([torch_parameter], lr=0.1,
                                            momentum=momentum, weight_decay=decay)
                for gradient in ([0.4, -0.3], [-0.2, 0.5], [0.8, 0.1]):
                    custom_parameter.grad = torch.tensor(gradient)
                    torch_parameter.grad = torch.tensor(gradient)
                    custom.step()
                    reference.step()
                    torch.testing.assert_close(custom_parameter, torch_parameter,
                                               rtol=0, atol=1e-7)

    def test_invalid_hyperparameters(self):
        parameter = torch.nn.Parameter(torch.ones(1))
        for kwargs in [{"lr": 0}, {"momentum": 1}, {"weight_decay": -1}]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                MomentumSGD([parameter], **kwargs)


if __name__ == "__main__":
    unittest.main()
