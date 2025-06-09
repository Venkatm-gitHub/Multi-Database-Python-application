import unittest
from unittest.mock import patch
from driver_loader import DatabaseDriverLoader
from exceptions import DriverNotAvailableException


class TestDatabaseDriverLoader(unittest.TestCase):

    def test_load_driver_oracle(self):
        with patch('importlib.import_module', return_value='mock_oracle'):
            loader = DatabaseDriverLoader()
            driver = loader.load_driver('oracle')
            self.assertEqual(driver, 'mock_oracle')

    def test_load_driver_snowflake(self):
        with patch('importlib.import_module', return_value='mock_snowflake'):
            loader = DatabaseDriverLoader()
            driver = loader.load_driver('snowflake')
            self.assertEqual(driver, 'mock_snowflake')

    def test_load_driver_invalid(self):
        loader = DatabaseDriverLoader()
        with self.assertRaises(DriverNotAvailableException):
            loader.load_driver('mysql')


if __name__ == '__main__':
    unittest.main()
