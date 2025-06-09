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
