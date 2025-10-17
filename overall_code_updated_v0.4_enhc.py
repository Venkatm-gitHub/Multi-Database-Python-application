# modified in 1) read_as_tuple, 2) get_connection - addeded 2 sub functions _is_connection_valid and _create_new_connection

# exceptions.py
from typing import Optional, Dict, Any
import logging

class DatabaseException(Exception):
    """Base exception for all database-related errors"""
    def __init__(self, message: str, error_code: Optional[str] = None, original_exception: Optional[Exception] = None):
        self.message = message
        self.error_code = error_code
        self.original_exception = original_exception
        super().__init__(self.message)

class ConnectionException(DatabaseException): pass
class AuthenticationException(DatabaseException): pass
class TableNotFoundException(DatabaseException): pass
class TableLockedException(DatabaseException): pass
class InvalidSQLException(DatabaseException): pass
class PermissionDeniedException(DatabaseException): pass
class TimeoutException(DatabaseException): pass
class TransactionException(DatabaseException): pass
class ConfigurationException(DatabaseException): pass
class DriverNotAvailableException(DatabaseException): pass

class DatabaseErrorHandler:
    """Standardizes error handling across different database implementations"""
    
    def __init__(self, db_type: str):
        self.db_type = db_type
        self.logger = logging.getLogger(f"DatabaseErrorHandler.{db_type}")
    
    def handle_exception(self, original_exception: Exception) -> DatabaseException:
        """Convert database-specific exceptions to standardized ones"""
        self.logger.error(f"Database error in {self.db_type}: {str(original_exception)}")
        
        if self.db_type == 'oracle':
            return self._handle_oracle_exception(original_exception)
        elif self.db_type == 'snowflake':
            return self._handle_snowflake_exception(original_exception)
        else:
            return DatabaseException(f"Unknown database error: {str(original_exception)}", 
                                   original_exception=original_exception)
    
    def _handle_oracle_exception(self, exc: Exception) -> DatabaseException:
        """Handle Oracle-specific exceptions"""
        exc_str = str(exc).lower()
        if 'ora-00942' in exc_str:
            return TableNotFoundException("Table or view does not exist", "ORA-00942", exc)
        elif 'ora-00054' in exc_str:
            return TableLockedException("Resource busy and acquire with NOWAIT specified", "ORA-00054", exc)
        elif 'ora-01017' in exc_str:
            return AuthenticationException("Invalid username/password", "ORA-01017", exc)
        elif 'ora-12170' in exc_str or 'ora-12541' in exc_str:
            return ConnectionException("TNS connection error", original_exception=exc)
        else:
            return DatabaseException(f"Oracle error: {str(exc)}", original_exception=exc)
    
    def _handle_snowflake_exception(self, exc: Exception) -> DatabaseException:
        """Handle Snowflake-specific exceptions"""
        exc_str = str(exc).lower()
        if 'does not exist' in exc_str:
            return TableNotFoundException("Object does not exist", original_exception=exc)
        elif 'insufficient privileges' in exc_str:
            return PermissionDeniedException("Insufficient privileges", original_exception=exc)
        elif 'authentication failed' in exc_str:
            return AuthenticationException("Authentication failed", original_exception=exc)
        else:
            return DatabaseException(f"Snowflake error: {str(exc)}", original_exception=exc)

#############################################################################################################

# interfaces.py
from abc import ABC, abstractmethod
from typing import Any, List, Tuple, Optional, Dict
import pandas as pd

class IDatabaseConfigBuilder(ABC):
    @abstractmethod
    def build(self) -> 'DatabaseConfig': pass

class IConnectionPool(ABC):
    @abstractmethod
    def get_connection(self) -> Any: pass
    
    @abstractmethod
    def return_connection(self, conn: Any) -> None: pass
    
    @abstractmethod
    def close_all(self) -> None: pass

class IConnectionStrategy(ABC):
    @abstractmethod
    def create_pool(self, config: 'DatabaseConfig') -> IConnectionPool: pass

class IDatabaseReader(ABC):
    @abstractmethod
    def read_as_tuple(self, sql: str, parameters: Optional[Dict] = None) -> Tuple[List[Tuple], int]: pass
    
    @abstractmethod
    def read_as_dataframe(self, sql: str, parameters: Optional[Dict] = None) -> Tuple[pd.DataFrame, int]: pass

class IDatabaseWriter(ABC):
    @abstractmethod
    def execute_write(self, sql: str, parameters: Optional[Dict] = None) -> int: pass
    
    @abstractmethod
    def execute_batch(self, sql: str, parameters_list: List[Dict]) -> int: pass

class ITransactionManager(ABC):
    @abstractmethod
    def begin_transaction(self) -> None: pass
    
    @abstractmethod
    def commit_transaction(self) -> None: pass
    
    @abstractmethod
    def rollback_transaction(self) -> None: pass

class IDatabaseSession(ABC):
    @abstractmethod
    def close(self) -> None: pass
    
    @property
    @abstractmethod
    def reader(self) -> IDatabaseReader: pass
    
    @property
    @abstractmethod
    def writer(self) -> IDatabaseWriter: pass
    
    @property
    @abstractmethod
    def transaction_manager(self) -> ITransactionManager: pass

class IConfigurationProvider(ABC):
    @abstractmethod
    def get_config(self, db_type: str) -> 'DatabaseConfig': pass

class IDatabaseDriverLoader(ABC):
    @abstractmethod
    def load_driver(self, db_type: str) -> Any: pass

#############################################################################################################

# configuration.py
import os
import time
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from interfaces import IConfigurationProvider, IDatabaseConfigBuilder

@dataclass
class DatabaseConfig:
    """Unified configuration for all database types"""
    db_type: str
    connection_params: Dict[str, Any] = field(default_factory=dict)
    pool_settings: Dict[str, Any] = field(default_factory=dict)
    query_settings: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self) -> None:
        """Validate configuration based on database type"""
        validator = ConfigValidator.get_validator(self.db_type)
        validator.validate(self)

class ConfigValidator:
    """Configuration validation for different database types"""
    
    @staticmethod
    def get_validator(db_type: str) -> 'ConfigValidator':
        if db_type == 'oracle':
            return OracleConfigValidator()
        elif db_type == 'snowflake':
            return SnowflakeConfigValidator()
        else:
            raise ConfigurationException(f"No validator for database type: {db_type}")
    
    def validate(self, config: DatabaseConfig) -> None:
        raise NotImplementedError

class OracleConfigValidator(ConfigValidator):
    def validate(self, config: DatabaseConfig) -> None:
        required_fields = ['host', 'port', 'service_name', 'username', 'password']
        for field in required_fields:
            if field not in config.connection_params or not config.connection_params[field]:
                raise ConfigurationException(f"Missing required Oracle configuration field: {field}")

class SnowflakeConfigValidator(ConfigValidator):
    def validate(self, config: DatabaseConfig) -> None:
        required_fields = ['account', 'user', 'password', 'warehouse', 'database']
        for field in required_fields:
            if field not in config.connection_params or not config.connection_params[field]:
                raise ConfigurationException(f"Missing required Snowflake configuration field: {field}")

# Configuration Builder Pattern Implementation
class DatabaseConfigBuilder(IDatabaseConfigBuilder):
    """Generic database configuration builder"""
    
    def __init__(self, db_type: str):
        self.config = DatabaseConfig(db_type=db_type)
    
    def with_connection_param(self, key: str, value: Any) -> 'DatabaseConfigBuilder':
        """Add a connection parameter"""
        self.config.connection_params[key] = value
        return self
    
    def with_connection_params(self, params: Dict[str, Any]) -> 'DatabaseConfigBuilder':
        """Add multiple connection parameters"""
        self.config.connection_params.update(params)
        return self
    
    def with_pool_setting(self, key: str, value: Any) -> 'DatabaseConfigBuilder':
        """Add a pool setting"""
        self.config.pool_settings[key] = value
        return self
    
    def with_pool_settings(self, settings: Dict[str, Any]) -> 'DatabaseConfigBuilder':
        """Add multiple pool settings"""
        self.config.pool_settings.update(settings)
        return self
    
    def with_query_setting(self, key: str, value: Any) -> 'DatabaseConfigBuilder':
        """Add a query setting"""
        self.config.query_settings[key] = value
        return self
    
    def with_query_settings(self, settings: Dict[str, Any]) -> 'DatabaseConfigBuilder':
        """Add multiple query settings"""
        self.config.query_settings.update(settings)
        return self
    
    def build(self) -> DatabaseConfig:
        """Build and validate the configuration"""
        self.config.validate()
        return self.config

class OracleConfigBuilder(DatabaseConfigBuilder):
    """Oracle-specific configuration builder with convenience methods"""
    
    def __init__(self):
        super().__init__('oracle')
    
    def with_host(self, host: str) -> 'OracleConfigBuilder':
        return self.with_connection_param('host', host)
    
    def with_port(self, port: int) -> 'OracleConfigBuilder':
        return self.with_connection_param('port', port)
    
    def with_service_name(self, service_name: str) -> 'OracleConfigBuilder':
        return self.with_connection_param('service_name', service_name)
    
    def with_credentials(self, username: str, password: str) -> 'OracleConfigBuilder':
        return (self.with_connection_param('username', username)
                   .with_connection_param('password', password))
    
    def with_pool_size(self, min_size: int = 1, max_size: int = 10, increment: int = 1) -> 'OracleConfigBuilder':
        return (self.with_pool_setting('min', min_size)
                   .with_pool_setting('max', max_size)
                   .with_pool_setting('increment', increment))
    
    def with_fetch_settings(self, fetch_size: int = 10000, batch_size: int = 1000) -> 'OracleConfigBuilder':
        return (self.with_query_setting('fetch_size', fetch_size)
                   .with_query_setting('batch_size', batch_size))

class SnowflakeConfigBuilder(DatabaseConfigBuilder):
    """Snowflake-specific configuration builder with convenience methods"""
    
    def __init__(self):
        super().__init__('snowflake')
    
    def with_account(self, account: str) -> 'SnowflakeConfigBuilder':
        return self.with_connection_param('account', account)
    
    def with_credentials(self, user: str, password: str) -> 'SnowflakeConfigBuilder':
        return (self.with_connection_param('user', user)
                   .with_connection_param('password', password))
    
    def with_warehouse(self, warehouse: str) -> 'SnowflakeConfigBuilder':
        return self.with_connection_param('warehouse', warehouse)
    
    def with_database(self, database: str, schema: str = 'PUBLIC') -> 'SnowflakeConfigBuilder':
        return (self.with_connection_param('database', database)
                   .with_connection_param('schema', schema))

class EnvironmentConfigurationProvider(IConfigurationProvider):
    """Provides configuration from environment variables"""
    
    def get_config(self, db_type: str) -> DatabaseConfig:
        if db_type == 'oracle':
            return self._get_oracle_config()
        elif db_type == 'snowflake':
            return self._get_snowflake_config()
        else:
            raise ConfigurationException(f"Unsupported database type: {db_type}")
    
    def _get_oracle_config(self) -> DatabaseConfig:
        return (OracleConfigBuilder()
                .with_host(os.getenv('ORACLE_HOST', 'localhost'))
                .with_port(int(os.getenv('ORACLE_PORT', '1521')))
                .with_service_name(os.getenv('ORACLE_SERVICE_NAME', ''))
                .with_credentials(
                    os.getenv('ORACLE_USERNAME', ''),
                    os.getenv('ORACLE_PASSWORD', '')
                )
                .with_pool_size(
                    int(os.getenv('ORACLE_POOL_MIN', '1')),
                    int(os.getenv('ORACLE_POOL_MAX', '10')),
                    int(os.getenv('ORACLE_POOL_INCREMENT', '1'))
                )
                .with_fetch_settings(
                    int(os.getenv('ORACLE_FETCH_SIZE', '10000')),
                    int(os.getenv('ORACLE_BATCH_SIZE', '1000'))
                )
                .build())
    
    def _get_snowflake_config(self) -> DatabaseConfig:
        return (SnowflakeConfigBuilder()
                .with_account(os.getenv('SNOWFLAKE_ACCOUNT', ''))
                .with_credentials(
                    os.getenv('SNOWFLAKE_USER', ''),
                    os.getenv('SNOWFLAKE_PASSWORD', '')
                )
                .with_warehouse(os.getenv('SNOWFLAKE_WAREHOUSE', ''))
                .with_database(
                    os.getenv('SNOWFLAKE_DATABASE', ''),
                    os.getenv('SNOWFLAKE_SCHEMA', 'PUBLIC')
                )
                .build())

#############################################################################################################

# driver_loader.py
import importlib
from typing import Any, Dict
from interfaces import IDatabaseDriverLoader

class DatabaseDriverLoader(IDatabaseDriverLoader):
    """Dynamically loads database drivers to reduce tight coupling"""
    
    _drivers: Dict[str, Any] = {}
    
    def load_driver(self, db_type: str) -> Any:
        if db_type in self._drivers:
            return self._drivers[db_type]
        
        try:
            if db_type == 'oracle':
                driver = importlib.import_module('cx_Oracle')
            elif db_type == 'snowflake':
                driver = importlib.import_module('snowflake.connector')
            else:
                raise DriverNotAvailableException(f"No driver mapping for database type: {db_type}")
            
            self._drivers[db_type] = driver
            return driver
            
        except ImportError as e:
            raise DriverNotAvailableException(f"Driver for {db_type} not available: {str(e)}")

#############################################################################################################

# connection_management.py
import threading
import time
from typing import Any, List
from queue import Queue, Empty
from interfaces import IConnectionPool, IConnectionStrategy

class BasicConnectionPool(IConnectionPool):
    """Thread-safe connection pool implementation"""
    
    def __init__(self, connection_factory, min_size: int = 1, max_size: int = 10):
        self.connection_factory = connection_factory
        self.min_size = min_size
        self.max_size = max_size
        self.pool = Queue(maxsize=max_size)
        self.active_connections = 0
        self.lock = threading.Lock()
        
        # Initialize minimum connections
        for _ in range(min_size):
            conn = self.connection_factory()
            self.pool.put(conn)
            self.active_connections += 1

    # 20251017 -modified    
    def get_connection(self) -> Any:
        try:
            conn = self.pool.get_nowait()
            #  Validate connection is alive
            if self._is_connection_valid(conn):
                return conn
            else:
                # Connection is stale, create new one
                self.logger.warning("Stale connection detected, creating new one")
                return self._create_new_connection()
        except Empty:
            with self.lock:
                if self.active_connections < self.max_size:
                    # Create new connection
                    conn = self.connection_factory()
                    self.active_connections += 1
                    return conn
                else:
                    # Wait for available connection
                    return self.pool.get()
                
    def _is_connection_valid(self, conn: Any) -> bool:
        """Check if connection is still alive"""
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM DUAL")  # Oracle ping , can use  conn.ping()  also
            cursor.fetchone()
            cursor.close()
            return True
        except:
            try:
                conn.close()  # Close stale connection
            except:
                pass
            self.active_connections -= 1
            return False
    
    def _create_new_connection(self) -> Any:
        """Create a new validated connection"""
        with self.lock:
            if self.active_connections < self.max_size:
                conn = self.connection_factory()
                self.active_connections += 1
                return conn
            else:
                raise ConnectionException("Max connections reached")
    
    def return_connection(self, conn: Any) -> None:
        if conn:
            self.pool.put(conn)
    
    def close_all(self) -> None:
        while not self.pool.empty():
            try:
                conn = self.pool.get_nowait()
                if hasattr(conn, 'close'):
                    conn.close()
            except Empty:
                break
        self.active_connections = 0

class OracleConnectionStrategy(IConnectionStrategy):
    def __init__(self, driver_loader: IDatabaseDriverLoader):
        self.driver_loader = driver_loader
    
    def create_pool(self, config: DatabaseConfig) -> IConnectionPool:
        driver = self.driver_loader.load_driver('oracle')
        
        def connection_factory():
            return driver.connect(
                user=config.connection_params['username'],
                password=config.connection_params['password'],
                dsn=driver.makedsn(
                    config.connection_params['host'],
                    config.connection_params['port'],
                    service_name=config.connection_params['service_name']
                )
            )
        
        return BasicConnectionPool(
            connection_factory,
            config.pool_settings.get('min', 1),
            config.pool_settings.get('max', 10)
        )

class SnowflakeConnectionStrategy(IConnectionStrategy):
    def __init__(self, driver_loader: IDatabaseDriverLoader):
        self.driver_loader = driver_loader
    
    def create_pool(self, config: DatabaseConfig) -> IConnectionPool:
        driver = self.driver_loader.load_driver('snowflake')
        
        def connection_factory():
            return driver.connect(**config.connection_params)
        
        return BasicConnectionPool(connection_factory, 1, 5)  # Snowflake typically uses fewer connections

#############################################################################################################

# database_operations.py

import logging
import time
from typing import Tuple, List, Dict, Any, Optional
import pandas as pd
from interfaces import IDatabaseReader, IDatabaseWriter, ITransactionManager

class QueryTimer:
    """Context manager for timing query execution"""
    
    def __init__(self, logger: logging.Logger, operation: str, sql: str = ""):
        self.logger = logger
        self.operation = operation
        self.sql = sql[:100] + "..." if len(sql) > 100 else sql  # Truncate long SQL
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            execution_time = time.perf_counter() - self.start_time
            if exc_type:
                self.logger.warning(f"{self.operation} failed after {execution_time:.3f}s - SQL: {self.sql}")
            else:
                self.logger.info(f"{self.operation} completed in {execution_time:.3f}s - SQL: {self.sql}")

class DatabaseReader(IDatabaseReader):
    """Generic database reader with error handling and timing"""
    
    def __init__(self, connection_pool: IConnectionPool, error_handler: DatabaseErrorHandler, config: DatabaseConfig):
        self.connection_pool = connection_pool
        self.error_handler = error_handler
        self.config = config
        self.logger = logging.getLogger(f"DatabaseReader.{config.db_type}")
    
    # 20251017 -modified
    def read_as_tuple(self, sql: str, parameters: Optional[Dict] = None) -> Tuple[List[Tuple], int]:
        conn = None
        cursor = None  # Initialize outside try
        with QueryTimer(self.logger, "READ_TUPLE", sql):
            try:
                conn = self.connection_pool.get_connection()
                cursor = conn.cursor()
                
                if parameters:
                    cursor.execute(sql, parameters)
                else:
                    cursor.execute(sql)
                
                fetch_size = self.config.query_settings.get('fetch_size')
                if fetch_size and hasattr(cursor, 'arraysize'):
                    cursor.arraysize = fetch_size
                
                data = cursor.fetchall()
                self.logger.info(f"Successfully read {len(data)} rows")
                return data, len(data)
                
            except Exception as e:
                self.logger.error(f"Error executing query: {sql}")
                raise self.error_handler.handle_exception(e)
            finally:
                # ✅ Always cleanup cursor first
                if cursor:
                    try:
                        cursor.close()
                    except:
                        pass
                # ✅ Then return connection
                if conn:
                    self.connection_pool.return_connection(conn)
        
    def read_as_dataframe(self, sql: str, parameters: Optional[Dict] = None) -> Tuple[pd.DataFrame, int]:
        conn = None
        with QueryTimer(self.logger, "READ_DATAFRAME", sql):
            try:
                conn = self.connection_pool.get_connection()
                cursor = conn.cursor()
                
                if parameters:
                    cursor.execute(sql, parameters)
                else:
                    cursor.execute(sql)
                
                # Get column names
                columns = [desc[0] for desc in cursor.description] if cursor.description else []
                
                # Set fetch size if configured
                fetch_size = self.config.query_settings.get('fetch_size')
                if fetch_size and hasattr(cursor, 'arraysize'):
                    cursor.arraysize = fetch_size
                
                data = cursor.fetchall()
                cursor.close()
                
                df = pd.DataFrame(data, columns=columns)
                self.logger.info(f"Successfully created DataFrame with {len(df)} rows")
                return df, len(df)
                
            except Exception as e:
                self.logger.error(f"Error executing query: {sql}")
                raise self.error_handler.handle_exception(e)
            finally:
                if conn:
                    self.connection_pool.return_connection(conn)

class DatabaseWriter(IDatabaseWriter):
    """Generic database writer with error handling and timing"""
    
    def __init__(self, connection_pool: IConnectionPool, error_handler: DatabaseErrorHandler, config: DatabaseConfig):
        self.connection_pool = connection_pool
        self.error_handler = error_handler
        self.config = config
        self.logger = logging.getLogger(f"DatabaseWriter.{config.db_type}")
    
    def execute_write(self, sql: str, parameters: Optional[Dict] = None) -> int:
        conn = None
        with QueryTimer(self.logger, "WRITE", sql):
            try:
                conn = self.connection_pool.get_connection()
                cursor = conn.cursor()
                
                if parameters:
                    cursor.execute(sql, parameters)
                else:
                    cursor.execute(sql)
                
                conn.commit()
                rowcount = cursor.rowcount
                cursor.close()
                
                self.logger.info(f"Successfully executed write operation, {rowcount} rows affected")
                return rowcount
                
            except Exception as e:
                if conn:
                    conn.rollback()
                self.logger.error(f"Error executing write: {sql}")
                raise self.error_handler.handle_exception(e)
            finally:
                if conn:
                    self.connection_pool.return_connection(conn)
    
    def execute_batch(self, sql: str, parameters_list: List[Dict]) -> int:
        conn = None
        with QueryTimer(self.logger, f"BATCH_WRITE ({len(parameters_list)} records)", sql):
            try:
                conn = self.connection_pool.get_connection()
                cursor = conn.cursor()
                
                # Process in batches if batch_size is configured
                batch_size = self.config.query_settings.get('batch_size', len(parameters_list))
                total_rowcount = 0
                
                for i in range(0, len(parameters_list), batch_size):
                    batch = parameters_list[i:i + batch_size]
                    cursor.executemany(sql, batch)
                    total_rowcount += cursor.rowcount
                
                conn.commit()
                cursor.close()
                
                self.logger.info(f"Successfully executed batch operation, {total_rowcount} rows affected")
                return total_rowcount
                
            except Exception as e:
                if conn:
                    conn.rollback()
                self.logger.error(f"Error executing batch: {sql}")
                raise self.error_handler.handle_exception(e)
            finally:
                if conn:
                    self.connection_pool.return_connection(conn)

class TransactionManager(ITransactionManager):
    """Generic transaction manager with context support and timing"""
    
    def __init__(self, connection_pool: IConnectionPool, error_handler: DatabaseErrorHandler):
        self.connection_pool = connection_pool
        self.error_handler = error_handler
        self.connection = None
        self.in_transaction = False
        self.transaction_start_time = None
        self.logger = logging.getLogger("TransactionManager")
    
    def begin_transaction(self) -> None:
        try:
            if self.in_transaction:
                raise TransactionException("Transaction already in progress")
            
            self.connection = self.connection_pool.get_connection()
            self.connection.autocommit = False
            self.in_transaction = True
            self.transaction_start_time = time.perf_counter()
            self.logger.info("Transaction started")
            
        except Exception as e:
            raise self.error_handler.handle_exception(e)
    
    def commit_transaction(self) -> None:
        try:
            if not self.in_transaction:
                raise TransactionException("No transaction in progress")
            
            self.connection.commit()
            
            if self.transaction_start_time:
                duration = time.perf_counter() - self.transaction_start_time
                self.logger.info(f"Transaction committed successfully in {duration:.3f}s")
            else:
                self.logger.info("Transaction committed")
            
        except Exception as e:
            raise self.error_handler.handle_exception(e)
        finally:
            self._cleanup_transaction()
    
    def rollback_transaction(self) -> None:
        try:
            if not self.in_transaction:
                raise TransactionException("No transaction in progress")
            
            self.connection.rollback()
            
            if self.transaction_start_time:
                duration = time.perf_counter() - self.transaction_start_time
                self.logger.info(f"Transaction rolled back after {duration:.3f}s")
            else:
                self.logger.info("Transaction rolled back")
            
        except Exception as e:
            raise self.error_handler.handle_exception(e)
        finally:
            self._cleanup_transaction()
    
    def _cleanup_transaction(self) -> None:
        if self.connection:
            self.connection_pool.return_connection(self.connection)
            self.connection = None
        self.in_transaction = False
        self.transaction_start_time = None

class TransactionContext:
    """Context manager for transactions with timing"""
    
    def __init__(self, transaction_manager: ITransactionManager):
        self.transaction_manager = transaction_manager
    
    def __enter__(self) -> ITransactionManager:
        self.transaction_manager.begin_transaction()
        return self.transaction_manager
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_type:
            self.transaction_manager.rollback_transaction()
        else:
            self.transaction_manager.commit_transaction()

#############################################################################################################

# session.py
from interfaces import IDatabaseSession

class DatabaseSession(IDatabaseSession):
    """Database session with resource management"""
    
    def __init__(self, reader: IDatabaseReader, writer: IDatabaseWriter, 
                 transaction_manager: ITransactionManager, connection_pool: IConnectionPool):
        self._reader = reader
        self._writer = writer
        self._transaction_manager = transaction_manager
        self._connection_pool = connection_pool
        self._closed = False
    
    @property
    def reader(self) -> IDatabaseReader:
        self._check_not_closed()
        return self._reader
    
    @property
    def writer(self) -> IDatabaseWriter:
        self._check_not_closed()
        return self._writer
    
    @property
    def transaction_manager(self) -> ITransactionManager:
        self._check_not_closed()
        return self._transaction_manager
    
    def transaction_context(self) -> TransactionContext:
        """Get a transaction context manager"""
        return TransactionContext(self._transaction_manager)
    
    def close(self) -> None:
        if not self._closed:
            self._connection_pool.close_all()
            self._closed = True
    
    def _check_not_closed(self) -> None:
        if self._closed:
            raise DatabaseException("Session is closed")
    
    def __enter__(self) -> 'DatabaseSession':
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

#############################################################################################################

# dependency_injection.py
from typing import Dict, Any, Callable, Type, TypeVar
import threading

T = TypeVar('T')

class DIContainer:
    """Simple dependency injection container"""
    
    def __init__(self):
        self._services: Dict[Type, Callable] = {}
        self._singletons: Dict[Type, Any] = {}
        self._lock = threading.Lock()
    
    def register_transient(self, interface: Type[T], implementation: Callable[[], T]) -> None:
        """Register a service as transient (new instance each time)"""
        self._services[interface] = implementation
    
    def register_singleton(self, interface: Type[T], implementation: Callable[[], T]) -> None:
        """Register a service as singleton (same instance always)"""
        self._services[interface] = implementation
        self._singletons[interface] = None
    
    def resolve(self, interface: Type[T]) -> T:
        """Resolve a service instance"""
        if interface not in self._services:
            raise ValueError(f"Service {interface} not registered")
        
        # Check if it's a singleton
        if interface in self._singletons:
            with self._lock:
                if self._singletons[interface] is None:
                    self._singletons[interface] = self._services[interface]()
                return self._singletons[interface]
        
        # Return new instance
        return self._services[interface]()

#############################################################################################################

# database_factory.py
from abc import ABC, abstractmethod
from typing import Dict

class AbstractDatabaseFactory(ABC):
    @abstractmethod
    def create_connection_strategy(self, driver_loader: IDatabaseDriverLoader) -> IConnectionStrategy: pass
    
    @abstractmethod
    def create_reader(self, connection_pool: IConnectionPool, error_handler: DatabaseErrorHandler, 
                     config: DatabaseConfig) -> IDatabaseReader: pass
    
    @abstractmethod
    def create_writer(self, connection_pool: IConnectionPool, error_handler: DatabaseErrorHandler, 
                     config: DatabaseConfig) -> IDatabaseWriter: pass
    
    @abstractmethod
    def create_transaction_manager(self, connection_pool: IConnectionPool, 
                                 error_handler: DatabaseErrorHandler) -> ITransactionManager: pass

class OracleFactory(AbstractDatabaseFactory):
    def create_connection_strategy(self, driver_loader: IDatabaseDriverLoader) -> IConnectionStrategy:
        return OracleConnectionStrategy(driver_loader)
    
    def create_reader(self, connection_pool: IConnectionPool, error_handler: DatabaseErrorHandler, 
                     config: DatabaseConfig) -> IDatabaseReader:
        return DatabaseReader(connection_pool, error_handler, config)
    
    def create_writer(self, connection_pool: IConnectionPool, error_handler: DatabaseErrorHandler, 
                     config: DatabaseConfig) -> IDatabaseWriter:
        return DatabaseWriter(connection_pool, error_handler, config)
    
    def create_transaction_manager(self, connection_pool: IConnectionPool, 
                                 error_handler: DatabaseErrorHandler) -> ITransactionManager:
        return TransactionManager(connection_pool, error_handler)

class SnowflakeFactory(AbstractDatabaseFactory):
    def create_connection_strategy(self, driver_loader: IDatabaseDriverLoader) -> IConnectionStrategy:
        return SnowflakeConnectionStrategy(driver_loader)
    
    def create_reader(self, connection_pool: IConnectionPool, error_handler: DatabaseErrorHandler, 
                     config: DatabaseConfig) -> IDatabaseReader:
        return DatabaseReader(connection_pool, error_handler, config)
    
    def create_writer(self, connection_pool: IConnectionPool, error_handler: DatabaseErrorHandler, 
                     config: DatabaseConfig) -> IDatabaseWriter:
        return DatabaseWriter(connection_pool, error_handler, config)
    
    def create_transaction_manager(self, connection_pool: IConnectionPool, 
                                 error_handler: DatabaseErrorHandler) -> ITransactionManager:
        return TransactionManager(connection_pool, error_handler)

class DatabaseFactoryRegistry:
    """Registry for database factories"""
    
    _factories: Dict[str, AbstractDatabaseFactory] = {
        'oracle': OracleFactory(),
        'snowflake': SnowflakeFactory()
    }
    
    @classmethod
    def register_factory(cls, db_type: str, factory: AbstractDatabaseFactory) -> None:
        cls._factories[db_type] = factory
    
    @classmethod
    def get_factory(cls, db_type: str) -> AbstractDatabaseFactory:
        if db_type not in cls._factories:
            raise ValueError(f"No factory registered for database type: {db_type}")
        return cls._factories[db_type]

#############################################################################################################

# service.py
import logging
from typing import Optional

class DatabaseService:
    """Main service for creating database sessions"""
    
    def __init__(self, config_provider: Optional[IConfigurationProvider] = None,
                 driver_loader: Optional[IDatabaseDriverLoader] = None):
        self.config_provider = config_provider or EnvironmentConfigurationProvider()
        self.driver_loader = driver_loader or DatabaseDriverLoader()
        self.logger = logging.getLogger("DatabaseService")
    
    def create_session(self, db_type: str) -> DatabaseSession:
        """Create a new database session"""
        try:
            # Get configuration and validate
            config = self.config_provider.get_config(db_type)
            config.validate()
            
            # Get factory and create components
            factory = DatabaseFactoryRegistry.get_factory(db_type)
            error_handler = DatabaseErrorHandler(db_type)
            
            # Create connection strategy and pool
            connection_strategy = factory.create_connection_strategy(self.driver_loader)
            connection_pool = connection_strategy.create_pool(config)
            
            # Create database operations
            reader = factory.create_reader(connection_pool, error_handler, config)
            writer = factory.create_writer(connection_pool, error_handler, config)
            transaction_manager = factory.create_transaction_manager(connection_pool, error_handler)
            
            session = DatabaseSession(reader, writer, transaction_manager, connection_pool)
            self.logger.info(f"Created database session for {db_type}")
            return session
            
        except Exception as e:
            self.logger.error(f"Failed to create session for {db_type}: {str(e)}")
            raise


#############################################################################################################

# main.py
import logging
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def setup_sample_environment():
    """Setup sample environment variables for testing"""
    os.environ.update({
        'ORACLE_HOST': 'localhost',
        'ORACLE_PORT': '1521',
        'ORACLE_SERVICE_NAME': 'XEPDB1',
        'ORACLE_USERNAME': 'hr',
        'ORACLE_PASSWORD': 'password',
        'SNOWFLAKE_ACCOUNT': 'your-account',
        'SNOWFLAKE_USER': 'your-user',
        'SNOWFLAKE_PASSWORD': 'your-password',
        'SNOWFLAKE_WAREHOUSE': 'COMPUTE_WH',
        'SNOWFLAKE_DATABASE': 'SAMPLE_DB',
    })

def run_config_builder_example():
    """Example of using configuration builders"""
    logger.info("Running Configuration Builder example...")
    
    # Oracle config using builder pattern
    oracle_config = (OracleConfigBuilder()
                    .with_host('localhost')
                    .with_port(1521)
                    .with_service_name('XEPDB1')
                    .with_credentials('hr', 'password')
                    .with_pool_size(min_size=2, max_size=20)
                    .with_fetch_settings(fetch_size=5000, batch_size=500)
                    .build())
    
    logger.info(f"Built Oracle config: {oracle_config.db_type}")
    logger.info(f"Connection params: {oracle_config.connection_params}")
    logger.info(f"Pool settings: {oracle_config.pool_settings}")
    
    # Snowflake config using builder pattern
    snowflake_config = (SnowflakeConfigBuilder()
                       .with_account('your-account')
                       .with_credentials('user', 'pass')
                       .with_warehouse('COMPUTE_WH')
                       .with_database('SAMPLE_DB', 'PUBLIC')
                       .build())
    
    logger.info(f"Built Snowflake config: {snowflake_config.db_type}")
    logger.info(f"Connection params: {snowflake_config.connection_params}")

def run_oracle_example():
    """Example Oracle usage with timing"""
    logger.info("Running Oracle example...")
    
    try:
        service = DatabaseService()
        
        with service.create_session('oracle') as session:
            # Simple read operation - timing will be logged automatically
            df, count = session.reader.read_as_dataframe("SELECT SYSDATE FROM DUAL")
            logger.info(f"Oracle query returned {count} rows")
            print(df)
            
            # Transaction example with timing
            with session.transaction_context() as tx:
                rows_affected = session.writer.execute_write(
                    "UPDATE employees SET salary = salary * 1.1 WHERE department_id = :dept_id",
                    {'dept_id': 10}
                )
                logger.info(f"Updated {rows_affected} employee records")
                
    except DriverNotAvailableException:
        logger.warning("Oracle driver not available - skipping Oracle example")
    except Exception as e:
        logger.error(f"Oracle example failed: {e}")

def run_snowflake_example():
    """Example Snowflake usage with timing"""
    logger.info("Running Snowflake example...")
    
    try:
        service = DatabaseService()
        
        with service.create_session('snowflake') as session:
            # Simple read operation - timing will be logged automatically
            df, count = session.reader.read_as_dataframe("SELECT CURRENT_TIMESTAMP() AS current_time")
            logger.info(f"Snowflake query returned {count} rows")
            print(df)
            
            # Example with parameters
            data, row_count = session.reader.read_as_tuple(
                "SELECT * FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_SCHEMA = %s LIMIT 5",
                ('PUBLIC',)
            )
            logger.info(f"Found {row_count} tables in PUBLIC schema")
            
            # Transaction example with timing
            with session.transaction_context() as tx:
                # Example: Create a temporary table and insert data
                session.writer.execute_write("""
                    CREATE OR REPLACE TEMPORARY TABLE temp_example (
                        id INTEGER,
                        name VARCHAR(50),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
                    )
                """)
                logger.info("Created temporary table")
                
                # Batch insert example - timing will show batch processing time
                sample_data = [
                    {'id': 1, 'name': 'Alice'},
                    {'id': 2, 'name': 'Bob'},
                    {'id': 3, 'name': 'Charlie'}
                ]
                
                rows_affected = session.writer.execute_batch(
                    "INSERT INTO temp_example (id, name) VALUES (%(id)s, %(name)s)",
                    sample_data
                )
                logger.info(f"Inserted {rows_affected} records into temporary table")
                
                # Read back the data
                result_df, result_count = session.reader.read_as_dataframe(
                    "SELECT * FROM temp_example ORDER BY id"
                )
                logger.info(f"Retrieved {result_count} records from temporary table")
                print("Inserted data:")
                print(result_df)
                
    except DriverNotAvailableException:
        logger.warning("Snowflake driver not available - skipping Snowflake example")
    except Exception as e:
        logger.error(f"Snowflake example failed: {e}")

if __name__ == "__main__":
    setup_sample_environment()
    
    # Demonstrate configuration builders
    run_config_builder_example()
    
    # Run database examples with timing
    run_oracle_example()
    run_snowflake_example()