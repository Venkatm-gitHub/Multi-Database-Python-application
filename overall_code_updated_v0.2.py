#exceptions.py

class DatabaseException(Exception):
    def __init__(self, message, error_code=None):
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)

class ConnectionException(DatabaseException): pass
class AuthenticationException(DatabaseException): pass
class TableNotFoundException(DatabaseException): pass
class TableLockedException(DatabaseException): pass
class InvalidSQLException(DatabaseException): pass
class PermissionDeniedException(DatabaseException): pass
class TimeoutException(DatabaseException): pass
class TransactionException(DatabaseException): pass



################################################################################################################

#interfaces.py


from abc import ABC, abstractmethod

class IDatabaseConfigBuilder(ABC):
    @abstractmethod
    def build(self): pass

class IConnectionStrategy(ABC):
    @abstractmethod
    def acquire(self): pass
    @abstractmethod
    def release(self, conn): pass

class IDatabaseReader(ABC):
    @abstractmethod
    def read_as_tuple(self, sql, parameters=None): pass
    @abstractmethod
    def read_as_dataframe(self, sql, parameters=None): pass

class IDatabaseWriter(ABC):
    @abstractmethod
    def execute_write(self, sql, parameters=None): pass
    @abstractmethod
    def execute_batch(self, sql, parameters_list): pass

class ITransactionManager(ABC):
    @abstractmethod
    def begin_transaction(self): pass
    @abstractmethod
    def commit_transaction(self): pass
    @abstractmethod
    def rollback_transaction(self): pass

class IDatabaseSession(ABC):
    @abstractmethod
    def close(self): pass
    @property
    @abstractmethod
    def reader(self): pass
    @property
    @abstractmethod
    def writer(self): pass
    @property
    @abstractmethod
    def transaction_manager(self): pass



################################################################################################################

#config_oracle.py

import os
from dataclasses import dataclass
from interfaces import IDatabaseConfigBuilder

@dataclass
class OracleConfig:
    host: str
    port: int
    service_name: str
    username: str
    password: str
    fetch_size: int = 10000
    batch_size: int = 1000
    pool_min: int = 1
    pool_max: int = 10

class OracleConfigBuilder(IDatabaseConfigBuilder):
    def build(self):
        return OracleConfig(
            host=os.getenv('ORACLE_HOST', 'localhost'),
            port=int(os.getenv('ORACLE_PORT', 1521)),
            service_name=os.getenv('ORACLE_SERVICE_NAME', ''),
            username=os.getenv('ORACLE_USERNAME', ''),
            password=os.getenv('ORACLE_PASSWORD', ''),
            fetch_size=int(os.getenv('ORACLE_FETCH_SIZE', 10000)),
            batch_size=int(os.getenv('ORACLE_BATCH_SIZE', 1000)),
            pool_min=int(os.getenv('ORACLE_POOL_MIN', 1)),
            pool_max=int(os.getenv('ORACLE_POOL_MAX', 10)),
        )


################################################################################################################

#config_snowflake.py

import os
from dataclasses import dataclass
from interfaces import IDatabaseConfigBuilder

@dataclass
class SnowflakeConfig:
    account: str
    user: str
    password: str
    warehouse: str
    database: str
    schema: str = "PUBLIC"

class SnowflakeConfigBuilder(IDatabaseConfigBuilder):
    def build(self):
        return SnowflakeConfig(
            account=os.getenv('SNOWFLAKE_ACCOUNT', ''),
            user=os.getenv('SNOWFLAKE_USER', ''),
            password=os.getenv('SNOWFLAKE_PASSWORD', ''),
            warehouse=os.getenv('SNOWFLAKE_WAREHOUSE', ''),
            database=os.getenv('SNOWFLAKE_DATABASE', ''),
            schema=os.getenv('SNOWFLAKE_SCHEMA', 'PUBLIC')
        )

################################################################################################################

#strategy_connection.py

from interfaces import IConnectionStrategy

class OracleConnectionStrategy(IConnectionStrategy):
    def __init__(self, config):
        import cx_Oracle
        self.config = config
        self.pool = cx_Oracle.SessionPool(
            user=config.username,
            password=config.password,
            dsn=cx_Oracle.makedsn(config.host, config.port, service_name=config.service_name),
            min=config.pool_min,
            max=config.pool_max,
            increment=1,
            threaded=True
        )
    def acquire(self):
        return self.pool.acquire()
    def release(self, conn):
        self.pool.release(conn)

class SnowflakeConnectionStrategy(IConnectionStrategy):
    def __init__(self, config):
        self.config = config
    def acquire(self):
        import snowflake.connector
        return snowflake.connector.connect(
            user=self.config.user,
            password=self.config.password,
            account=self.config.account,
            warehouse=self.config.warehouse,
            database=self.config.database,
            schema=self.config.schema
        )
    def release(self, conn):
        conn.close()


################################################################################################################

#database_factory.py

from abc import ABC, abstractmethod

class DatabaseFactoryRegistry:
    _factories = {}
    @classmethod
    def register_factory(cls, db_type, factory):
        cls._factories[db_type] = factory
    @classmethod
    def get_factory(cls, db_type):
        return cls._factories[db_type]

class AbstractDatabaseFactory(ABC):
    @abstractmethod
    def create_config_builder(self): pass
    @abstractmethod
    def create_connection_strategy(self, config): pass
    @abstractmethod
    def create_reader(self, conn, config): pass
    @abstractmethod
    def create_writer(self, conn, config): pass
    @abstractmethod
    def create_transaction_manager(self, conn): pass


################################################################################################################

#oracle_factory.py

from config_oracle import OracleConfigBuilder
from strategy_connection import OracleConnectionStrategy
from interfaces import IDatabaseReader, IDatabaseWriter, ITransactionManager
from database_factory import AbstractDatabaseFactory, DatabaseFactoryRegistry

class OracleReader(IDatabaseReader):
    def __init__(self, conn, config): self.conn, self.config = conn, config
    def read_as_tuple(self, sql, parameters=None):
        cur = self.conn.cursor()
        cur.execute(sql, parameters or {})
        data = cur.fetchall()
        cur.close()
        return data, len(data)
    def read_as_dataframe(self, sql, parameters=None):
        import pandas as pd
        cur = self.conn.cursor()
        cur.execute(sql, parameters or {})
        cols = [desc[0] for desc in cur.description]
        data = cur.fetchall()
        cur.close()
        return pd.DataFrame(data, columns=cols), len(data)

class OracleWriter(IDatabaseWriter):
    def __init__(self, conn, config): self.conn, self.config = conn, config
    def execute_write(self, sql, parameters=None):
        cur = self.conn.cursor()
        cur.execute(sql, parameters or {})
        self.conn.commit()
        rowcount = cur.rowcount
        cur.close()
        return rowcount
    def execute_batch(self, sql, parameters_list):
        cur = self.conn.cursor()
        cur.executemany(sql, parameters_list)
        self.conn.commit()
        rowcount = cur.rowcount
        cur.close()
        return rowcount

class OracleTransactionManager(ITransactionManager):
    def __init__(self, conn): self.conn = conn
    def begin_transaction(self): self.conn.autocommit = False
    def commit_transaction(self): self.conn.commit()
    def rollback_transaction(self): self.conn.rollback()

class OracleFactory(AbstractDatabaseFactory):
    def create_config_builder(self): return OracleConfigBuilder()
    def create_connection_strategy(self, config): return OracleConnectionStrategy(config)
    def create_reader(self, conn, config): return OracleReader(conn, config)
    def create_writer(self, conn, config): return OracleWriter(conn, config)
    def create_transaction_manager(self, conn): return OracleTransactionManager(conn)

DatabaseFactoryRegistry.register_factory('oracle', OracleFactory())



################################################################################################################


#snowflake_factory.py

from config_snowflake import SnowflakeConfigBuilder
from strategy_connection import SnowflakeConnectionStrategy
from interfaces import IDatabaseReader, IDatabaseWriter, ITransactionManager
from database_factory import AbstractDatabaseFactory, DatabaseFactoryRegistry

class SnowflakeReader(IDatabaseReader):
    def __init__(self, conn, config): self.conn, self.config = conn, config
    def read_as_tuple(self, sql, parameters=None):
        cur = self.conn.cursor()
        cur.execute(sql, parameters or {})
        data = cur.fetchall()
        cur.close()
        return data, len(data)
    def read_as_dataframe(self, sql, parameters=None):
        import pandas as pd
        cur = self.conn.cursor()
        cur.execute(sql, parameters or {})
        cols = [desc[0] for desc in cur.description]
        data = cur.fetchall()
        cur.close()
        return pd.DataFrame(data, columns=cols), len(data)

class SnowflakeWriter(IDatabaseWriter):
    def __init__(self, conn, config): self.conn, self.config = conn, config
    def execute_write(self, sql, parameters=None):
        cur = self.conn.cursor()
        cur.execute(sql, parameters or {})
        self.conn.commit()
        rowcount = cur.rowcount
        cur.close()
        return rowcount
    def execute_batch(self, sql, parameters_list):
        cur = self.conn.cursor()
        cur.executemany(sql, parameters_list)
        self.conn.commit()
        rowcount = cur.rowcount
        cur.close()
        return rowcount

class SnowflakeTransactionManager(ITransactionManager):
    def __init__(self, conn): self.conn = conn
    def begin_transaction(self): pass  # Snowflake autocommit by default
    def commit_transaction(self): self.conn.commit()
    def rollback_transaction(self): self.conn.rollback()

class SnowflakeFactory(AbstractDatabaseFactory):
    def create_config_builder(self): return SnowflakeConfigBuilder()
    def create_connection_strategy(self, config): return SnowflakeConnectionStrategy(config)
    def create_reader(self, conn, config): return SnowflakeReader(conn, config)
    def create_writer(self, conn, config): return SnowflakeWriter(conn, config)
    def create_transaction_manager(self, conn): return SnowflakeTransactionManager(conn)

DatabaseFactoryRegistry.register_factory('snowflake', SnowflakeFactory())


################################################################################################################


#session.py


from interfaces import IDatabaseSession

class DatabaseSession(IDatabaseSession):
    def __init__(self, conn, reader, writer, transaction_manager, connection_strategy):
        self._conn = conn
        self._reader = reader
        self._writer = writer
        self._transaction_manager = transaction_manager
        self._connection_strategy = connection_strategy

    @property
    def reader(self):
        return self._reader

    @property
    def writer(self):
        return self._writer

    @property
    def transaction_manager(self):
        return self._transaction_manager

    def close(self):
        self._connection_strategy.release(self._conn)


################################################################################################################

#service.py

from session import DatabaseSession

class DatabaseService:
    def __init__(self, factory):
        self._factory = factory

    def create_session(self):
        config = self._factory.create_config_builder().build()
        connection_strategy = self._factory.create_connection_strategy(config)
        conn = connection_strategy.acquire()
        reader = self._factory.create_reader(conn, config)
        writer = self._factory.create_writer(conn, config)
        transaction_manager = self._factory.create_transaction_manager(conn)
        return DatabaseSession(conn, reader, writer, transaction_manager, connection_strategy)


################################################################################################################

#main.py

import logging
from database_factory import DatabaseFactoryRegistry
from service import DatabaseService

logging.basicConfig(level=logging.INFO)

def run_oracle():
    factory = DatabaseFactoryRegistry.get_factory('oracle')
    service = DatabaseService(factory)
    session = service.create_session()
    try:
        df, count = session.reader.read_as_dataframe("SELECT * FROM employees")
        print(df)
        updated = session.writer.execute_write("UPDATE employees SET salary=salary+1 WHERE rownum<10")
        print(f"Rows updated: {updated}")
    finally:
        session.close()

def run_snowflake():
    factory = DatabaseFactoryRegistry.get_factory('snowflake')
    service = DatabaseService(factory)
    session = service.create_session()
    try:
        df, count = session.reader.read_as_dataframe("SELECT * FROM mytable")
        print(df)
    finally:
        session.close()

if __name__ == "__main__":
    run_oracle()
    run_snowflake()
