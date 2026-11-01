import unittest

from stochopt.samplers import RandomReshufflingSampler, ShuffleOnceSampler


class SamplerTests(unittest.TestCase):
    def test_each_epoch_visits_each_index_once(self):
        dataset = range(40)
        for sampler_class in (RandomReshufflingSampler, ShuffleOnceSampler):
            sampler = sampler_class(dataset, seed=7)
            for _ in range(3):
                self.assertEqual(sorted(sampler), list(dataset))

    def test_epoch_behavior_and_seed(self):
        dataset = range(40)
        reshuffled = RandomReshufflingSampler(dataset, seed=7)
        first, second = list(reshuffled), list(reshuffled)
        self.assertNotEqual(first, second)
        self.assertEqual(first, list(RandomReshufflingSampler(dataset, seed=7)))
        once = ShuffleOnceSampler(dataset, seed=7)
        self.assertEqual(list(once), list(once))


if __name__ == "__main__":
    unittest.main()
