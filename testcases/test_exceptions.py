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
