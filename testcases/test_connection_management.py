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
