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
