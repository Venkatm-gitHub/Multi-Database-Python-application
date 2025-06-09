import unittest
from unittest.mock import MagicMock, patch
from service import DatabaseService
from exceptions import DriverNotAvailableException


class TestDatabaseService(unittest.TestCase):

    @patch('service.EnvironmentConfigurationProvider')
    @patch('service.DatabaseDriverLoader')
    def test_create_session_valid(self, mock_loader_cls, mock_provider_cls):
        mock_provider = MagicMock()
        mock_loader = MagicMock()
        mock_provider_cls.return_value = mock_provider
        mock_loader_cls.return_value = mock_loader

        config = MagicMock()
        mock_provider.get_config.return_value = config
        config.validate.return_value = True

        service = DatabaseService()
        with patch('service.DatabaseFactoryRegistry.get_factory') as mock_get_factory:
            factory = MagicMock()
            mock_get_factory.return_value = factory
            factory.create_connection_strategy.return_value.create_pool.return_value = MagicMock()
            factory.create_reader.return_value = MagicMock()
            factory.create_writer.return_value = MagicMock()
            factory.create_transaction_manager.return_value = MagicMock()

            session = service.create_session("oracle")
            self.assertIsNotNone(session)

    def test_create_session_invalid_type(self):
        service = DatabaseService()
        with self.assertRaises(Exception):
            service.create_session("invalid-db")


if __name__ == '__main__':
    unittest.main()
