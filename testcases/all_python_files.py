# run_all_tests.py
#run_all_tests
import unittest
import os

def discover_and_run_tests(test_directory='.', pattern='test_*.py'):
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=test_directory, pattern=pattern)

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    if result.wasSuccessful():
        print("\n✅ All tests passed!")
        exit(0)
    else:
        print(f"\n❌ {len(result.failures)} test(s) failed, {len(result.errors)} error(s).")
        exit(1)

if __name__ == "__main__":
    print("🔍 Discovering and running tests...")
    discover_and_run_tests()
################################################
# test_configuration.py
import unittest
from configuration import (
    OracleConfigBuilder,
    SnowflakeConfigBuilder,
    ConfigurationException,
    EnvironmentConfigurationProvider
)
import os
from unittest.mock import patch


class TestOracleConfigBuilder(unittest.TestCase):

    def test_valid_oracle_config(self):
        config = (OracleConfigBuilder()
                  .with_host("localhost")
                  .with_port(1521)
                  .with_service_name("XEPDB1")
                  .with_credentials("user", "pass")
                  .with_pool_size(1, 5)
                  .with_fetch_settings(100, 10)
                  .build())
        self.assertEqual(config.db_type, "oracle")
        self.assertIn("host", config.connection_params)
        self.assertEqual(config.pool_settings['min'], 1)

    def test_missing_oracle_fields_raises(self):
        builder = OracleConfigBuilder().with_host("localhost")
        with self.assertRaises(ConfigurationException):
            builder.build()


class TestSnowflakeConfigBuilder(unittest.TestCase):

    def test_valid_snowflake_config(self):
        config = (SnowflakeConfigBuilder()
                  .with_account("acc")
                  .with_credentials("usr", "pwd")
                  .with_warehouse("WH")
                  .with_database("DB")
                  .build())
        self.assertEqual(config.db_type, "snowflake")
        self.assertIn("account", config.connection_params)

    def test_missing_snowflake_fields_raises(self):
        builder = SnowflakeConfigBuilder().with_account("acc")
        with self.assertRaises(ConfigurationException):
            builder.build()


class TestEnvironmentConfigurationProvider(unittest.TestCase):

    def test_get_oracle_config_from_env(self):
        env = {
            'ORACLE_HOST': 'localhost',
            'ORACLE_PORT': '1521',
            'ORACLE_SERVICE_NAME': 'XEPDB1',
            'ORACLE_USERNAME': 'user',
            'ORACLE_PASSWORD': 'pass'
        }
        with patch.dict(os.environ, env, clear=True):
            provider = EnvironmentConfigurationProvider()
            config = provider.get_config('oracle')
            self.assertEqual(config.connection_params['host'], 'localhost')

    def test_get_snowflake_config_from_env(self):
        env = {
            'SNOWFLAKE_ACCOUNT': 'acc',
            'SNOWFLAKE_USER': 'usr',
            'SNOWFLAKE_PASSWORD': 'pwd',
            'SNOWFLAKE_WAREHOUSE': 'WH',
            'SNOWFLAKE_DATABASE': 'DB'
        }
        with patch.dict(os.environ, env, clear=True):
            provider = EnvironmentConfigurationProvider()
            config = provider.get_config('snowflake')
            self.assertEqual(config.connection_params['account'], 'acc')

    def test_invalid_type_raises(self):
        provider = EnvironmentConfigurationProvider()
        with self.assertRaises(ConfigurationException):
            provider.get_config('mysql')


if __name__ == '__main__':
    unittest.main()
################################################
# test_connection_management.py
import unittest
from unittest.mock import MagicMock, patch
from connection_management import BasicConnectionPool
from exceptions import ConfigurationException


class TestBasicConnectionPool(unittest.TestCase):

    def test_connection_creation_and_reuse(self):
        factory = MagicMock()
        conn_instance = MagicMock()
        factory.return_value = conn_instance

        pool = BasicConnectionPool(factory, min_size=1, max_size=2)
        conn1 = pool.get_connection()
        pool.return_connection(conn1)
        conn2 = pool.get_connection()

        self.assertEqual(conn1, conn2)
        self.assertLessEqual(pool.active_connections, 2)

    def test_close_all(self):
        conn = MagicMock()
        conn.close = MagicMock()
        factory = MagicMock(return_value=conn)
        pool = BasicConnectionPool(factory, min_size=2, max_size=2)

        pool.close_all()
        self.assertEqual(pool.active_connections, 0)
        conn.close.assert_called()


if __name__ == '__main__':
    unittest.main()
################################################
# test_database_extended.py
import unittest
from unittest.mock import MagicMock, patch
from exceptions import DatabaseException, DatabaseErrorHandler
from database_operations import DatabaseReader, DatabaseWriter, TransactionManager
from connection_management import BasicConnectionPool
from configuration import DatabaseConfig

class TestDatabaseComponents(unittest.TestCase):

    def setUp(self):
        self.mock_conn = MagicMock()
        self.mock_cursor = MagicMock()
        self.mock_conn.cursor.return_value = self.mock_cursor
        self.mock_cursor.fetchall.return_value = [("row1",), ("row2",)]
        self.mock_cursor.description = [("col1",)]
        self.mock_cursor.rowcount = 2

        self.connection_pool = MagicMock(spec=BasicConnectionPool)
        self.connection_pool.get_connection.return_value = self.mock_conn
        self.error_handler = DatabaseErrorHandler("oracle")
        self.config = DatabaseConfig(db_type="oracle", query_settings={"fetch_size": 100})

    def test_read_as_dataframe(self):
        reader = DatabaseReader(self.connection_pool, self.error_handler, self.config)
        df, count = reader.read_as_dataframe("SELECT * FROM dummy")
        self.assertEqual(count, 2)
        self.assertFalse(df.empty)

    def test_execute_write(self):
        writer = DatabaseWriter(self.connection_pool, self.error_handler, self.config)
        count = writer.execute_write("UPDATE dummy SET x = 1")
        self.assertEqual(count, 2)
        self.mock_conn.commit.assert_called_once()

    def test_execute_batch(self):
        writer = DatabaseWriter(self.connection_pool, self.error_handler, self.config)
        self.mock_cursor.executemany = MagicMock()
        rows = writer.execute_batch("INSERT INTO dummy VALUES (%(id)s)", [{'id': 1}, {'id': 2}])
        self.assertEqual(rows, 2)

    def test_transaction_commit(self):
        tm = TransactionManager(self.connection_pool, self.error_handler)
        tm.begin_transaction()
        tm.commit_transaction()
        self.assertFalse(tm.in_transaction)
        self.mock_conn.commit.assert_called_once()

    def test_transaction_rollback(self):
        tm = TransactionManager(self.connection_pool, self.error_handler)
        tm.begin_transaction()
        tm.rollback_transaction()
        self.assertFalse(tm.in_transaction)
        self.mock_conn.rollback.assert_called_once()

    def test_transaction_nested_error(self):
        tm = TransactionManager(self.connection_pool, self.error_handler)
        tm.begin_transaction()
        with self.assertRaises(DatabaseException):
            tm.begin_transaction()

if __name__ == '__main__':
    unittest.main()
################################################
# test_database_factory.py
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
################################################
# test_database_operations.py
import unittest
from unittest.mock import MagicMock
from database_operations import DatabaseReader, DatabaseWriter, TransactionManager
from configuration import DatabaseConfig
from exceptions import DatabaseErrorHandler


class TestDatabaseReaderWriter(unittest.TestCase):

    def setUp(self):
        self.mock_conn = MagicMock()
        self.mock_cursor = MagicMock()
        self.mock_conn.cursor.return_value = self.mock_cursor
        self.mock_cursor.fetchall.return_value = [(1,), (2,)]
        self.mock_cursor.description = [("col1",)]
        self.mock_cursor.rowcount = 2

        self.connection_pool = MagicMock()
        self.connection_pool.get_connection.return_value = self.mock_conn
        self.error_handler = DatabaseErrorHandler("oracle")
        self.config = DatabaseConfig("oracle", query_settings={"fetch_size": 100, "batch_size": 2})

    def test_read_as_tuple(self):
        reader = DatabaseReader(self.connection_pool, self.error_handler, self.config)
        result, count = reader.read_as_tuple("SELECT * FROM DUAL")
        self.assertEqual(count, 2)

    def test_read_as_dataframe(self):
        reader = DatabaseReader(self.connection_pool, self.error_handler, self.config)
        df, count = reader.read_as_dataframe("SELECT * FROM DUAL")
        self.assertEqual(count, 2)
        self.assertFalse(df.empty)

    def test_execute_write(self):
        writer = DatabaseWriter(self.connection_pool, self.error_handler, self.config)
        count = writer.execute_write("UPDATE dummy SET x=1")
        self.assertEqual(count, 2)

    def test_execute_batch(self):
        writer = DatabaseWriter(self.connection_pool, self.error_handler, self.config)
        self.mock_cursor.executemany = MagicMock()
        rows = writer.execute_batch("INSERT INTO dummy VALUES (%(id)s)", [{'id': 1}, {'id': 2}])
        self.assertEqual(rows, 2)

    def test_transaction_commit_and_rollback(self):
        tm = TransactionManager(self.connection_pool, self.error_handler)
        tm.begin_transaction()
        tm.commit_transaction()
        tm.begin_transaction()
        tm.rollback_transaction()
        self.assertFalse(tm.in_transaction)


if __name__ == '__main__':
    unittest.main()
################################################
# test_dependency_injection.py
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
################################################
# test_driver_loader.py
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
################################################
# test_exceptions.py
import unittest
from exceptions import (
    DatabaseErrorHandler, TableNotFoundException, AuthenticationException,
    ConnectionException
)


class DummyOracleError(Exception):
    def __str__(self):
        return "ORA-00942: table or view does not exist"


class DummySnowflakeError(Exception):
    def __str__(self):
        return "Object does not exist"


class TestDatabaseErrorHandler(unittest.TestCase):

    def test_oracle_table_not_found(self):
        handler = DatabaseErrorHandler("oracle")
        exc = handler.handle_exception(DummyOracleError())
        self.assertIsInstance(exc, TableNotFoundException)

    def test_snowflake_table_not_found(self):
        handler = DatabaseErrorHandler("snowflake")
        exc = handler.handle_exception(DummySnowflakeError())
        self.assertIsInstance(exc, TableNotFoundException)

    def test_oracle_authentication_error(self):
        class AuthError(Exception):
            def __str__(self): return "ORA-01017: invalid username/password"

        handler = DatabaseErrorHandler("oracle")
        exc = handler.handle_exception(AuthError())
        self.assertIsInstance(exc, AuthenticationException)

    def test_oracle_connection_error(self):
        class ConnError(Exception):
            def __str__(self): return "ORA-12170: TNS timeout"

        handler = DatabaseErrorHandler("oracle")
        exc = handler.handle_exception(ConnError())
        self.assertIsInstance(exc, ConnectionException)


if __name__ == '__main__':
    unittest.main()
################################################
# test_service.py
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
################################################
# test_session.py
import unittest
from unittest.mock import MagicMock
from session import DatabaseSession
from exceptions import DatabaseException


class TestDatabaseSession(unittest.TestCase):

    def setUp(self):
        self.reader = MagicMock()
        self.writer = MagicMock()
        self.tx_manager = MagicMock()
        self.pool = MagicMock()
        self.session = DatabaseSession(self.reader, self.writer, self.tx_manager, self.pool)

    def test_accessors_work_when_open(self):
        self.assertEqual(self.session.reader, self.reader)
        self.assertEqual(self.session.writer, self.writer)
        self.assertEqual(self.session.transaction_manager, self.tx_manager)

    def test_close_session(self):
        self.session.close()
        with self.assertRaises(DatabaseException):
            _ = self.session.reader

    def test_context_manager(self):
        with DatabaseSession(self.reader, self.writer, self.tx_manager, self.pool) as sess:
            self.assertIs(sess.writer, self.writer)


if __name__ == '__main__':
    unittest.main()
################################################
