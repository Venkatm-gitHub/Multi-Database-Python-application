import unittest
from dependency_injection import DIContainer


class TestDIContainer(unittest.TestCase):

    def test_transient_registration(self):
        container = DIContainer()
        container.register_transient(str, lambda: "value")
        self.assertEqual(container.resolve(str), "value")
        self.assertNotEqual(container.resolve(str), container.resolve(str))  # new instance each time

    def test_singleton_registration(self):
        container = DIContainer()
        container.register_singleton(int, lambda: 42)
        self.assertEqual(container.resolve(int), 42)
        self.assertEqual(container.resolve(int), 42)  # same instance

    def test_unregistered_raises(self):
        container = DIContainer()
        with self.assertRaises(ValueError):
            container.resolve(list)


if __name__ == '__main__':
    unittest.main()
