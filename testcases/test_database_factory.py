import unittest
from database_factory import DatabaseFactoryRegistry
from exceptions import DatabaseException


class TestDatabaseFactoryRegistry(unittest.TestCase):

    def test_get_factory_success(self):
        factory = DatabaseFactoryRegistry.get_factory('oracle')
        self.assertIsNotNone(factory)
        self.assertTrue(hasattr(factory, 'create_connection_strategy'))

    def test_get_factory_invalid_type(self):
        with self.assertRaises(ValueError):
            DatabaseFactoryRegistry.get_factory('mysql')


if __name__ == '__main__':
    unittest.main()
