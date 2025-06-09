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
