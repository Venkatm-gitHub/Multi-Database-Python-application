#exceptions.py
python
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


######################################################################################################
#interfaces.py

from abc import ABC, abstractmethod

class IDatabaseConfigBuilder(ABC):
    @abstractmethod
    def build(self): pass

class IDatabaseConnection(ABC):
    @abstractmethod
    def connect(self): pass
    @abstractmethod
    def disconnect(self): pass
    @abstractmethod
    def is_connected(self): pass

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


######################################################################################################

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


######################################################################################################
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

######################################################################################################
#strategy_connection.py


from abc import ABC, abstractmethod

class IConnectionStrategy(ABC):
    @abstractmethod
    def acquire(self): pass
    @abstractmethod
    def release(self, conn): pass

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

######################################################################################################
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


######################################################################################################

#oracle_factory.py

from config_oracle import OracleConfigBuilder
from strategy_connection import OracleConnectionStrategy
from interfaces import IDatabaseReader, IDatabaseWriter, ITransactionManager

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

from database_factory import AbstractDatabaseFactory, DatabaseFactoryRegistry
class OracleFactory(AbstractDatabaseFactory):
    def create_config_builder(self): return OracleConfigBuilder()
    def create_connection_strategy(self, config): return OracleConnectionStrategy(config)
    def create_reader(self, conn, config): return OracleReader(conn, config)
    def create_writer(self, conn, config): return OracleWriter(conn, config)
    def create_transaction_manager(self, conn): return OracleTransactionManager(conn)

DatabaseFactoryRegistry.register_factory('oracle', OracleFactory())


######################################################################################################

#snowflake_factory.py

from config_snowflake import SnowflakeConfigBuilder
from strategy_connection import SnowflakeConnectionStrategy
from interfaces import IDatabaseReader, IDatabaseWriter, ITransactionManager

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
    def begin_transaction(self): pass # Snowflake autocommit by default
    def commit_transaction(self): self.conn.commit()
    def rollback_transaction(self): self.conn.rollback()

from database_factory import AbstractDatabaseFactory, DatabaseFactoryRegistry
class SnowflakeFactory(AbstractDatabaseFactory):
    def create_config_builder(self): return SnowflakeConfigBuilder()
    def create_connection_strategy(self, config): return SnowflakeConnectionStrategy(config)
    def create_reader(self, conn, config): return SnowflakeReader(conn, config)
    def create_writer(self, conn, config): return SnowflakeWriter(conn, config)
    def create_transaction_manager(self, conn): return SnowflakeTransactionManager(conn)

DatabaseFactoryRegistry.register_factory('snowflake', SnowflakeFactory())

######################################################################################################

#database_manager.py

import logging

class UniversalDatabaseManager:
    def __init__(self, db_type):
        from database_factory import DatabaseFactoryRegistry
        self.factory = DatabaseFactoryRegistry.get_factory(db_type)
        self.config = self.factory.create_config_builder().build()
        self.conn_strategy = self.factory.create_connection_strategy(self.config)
        self.conn = None
        self.logger = logging.getLogger(__name__)

    def __enter__(self):
        self.conn = self.conn_strategy.acquire()
        self.reader = self.factory.create_reader(self.conn, self.config)
        self.writer = self.factory.create_writer(self.conn, self.config)
        self.trans = self.factory.create_transaction_manager(self.conn)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.conn_strategy.release(self.conn)

    def execute_query(self, sql, params=None, as_dataframe=True):
        self.logger.info(f"Executing query: {sql}")
        if as_dataframe:
            return self.reader.read_as_dataframe(sql, params)
        return self.reader.read_as_tuple(sql, params)

    def execute_write(self, sql, params=None):
        self.logger.info(f"Executing write: {sql}")
        return self.writer.execute_write(sql, params)

    def execute_batch(self, sql, params_list):
        self.logger.info(f"Executing batch: {sql}")
        return self.writer.execute_batch(sql, params_list)



######################################################################################################

#example.py

import logging
from database_manager import UniversalDatabaseManager

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    # Oracle Example
    with UniversalDatabaseManager('oracle') as db:
        df, count = db.execute_query("SELECT * FROM employees", as_dataframe=True)
        print(df)
        updated = db.execute_write("UPDATE employees SET salary=salary+1 WHERE rownum<10")
        print(f"Rows updated: {updated}")

    # Snowflake Example
    with UniversalDatabaseManager('snowflake') as db:
        df, count = db.execute_query("SELECT * FROM mytable", as_dataframe=True)
        print(df)


######################################################################################################


