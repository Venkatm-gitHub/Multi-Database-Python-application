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
