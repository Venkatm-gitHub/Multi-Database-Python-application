def read_as_dataframe_enforced_schema(self, sql: str, parameters: Optional[Dict] = None) -> Tuple[pd.DataFrame, int]:
    """
    Reads Oracle data into a DataFrame and enforces Oracle-native types using the database cursor metadata.
    If a column's Oracle type is not mapped, defaults that column to string and logs the event.
    """
    conn = None
    with QueryTimer(self.logger, "READ_DF_SCHEMA", sql):
        try:
            conn = self.connection_pool.get_connection()
            cursor = conn.cursor()
            if parameters:
                cursor.execute(sql, parameters)
            else:
                cursor.execute(sql)

            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            oracle_types = [desc[1] for desc in cursor.description] if cursor.description else []

            fetch_size = self.config.query_settings.get('fetch_size')
            if fetch_size and hasattr(cursor, 'arraysize'):
                cursor.arraysize = fetch_size

            data = cursor.fetchall()
            cursor.close()

            df = pd.DataFrame(data, columns=columns)
            # Only apply schema mapping for Oracle
            if self.config.db_type == 'oracle':
                try:
                    import cx_Oracle
                    type_map = {
                        cx_Oracle.DB_TYPE_NUMBER: float,
                        cx_Oracle.DB_TYPE_VARCHAR: str,
                        cx_Oracle.DB_TYPE_CHAR: str,
                        cx_Oracle.DB_TYPE_DATE: 'datetime64[ns]',
                        cx_Oracle.DB_TYPE_TIMESTAMP: 'datetime64[ns]',
                        cx_Oracle.DB_TYPE_TIMESTAMP_T[col] = pd.to_datetime(df[col], errors='coerce')
                            else:
                                df[col] = df[col].astype(target_type, errors='ignore')
                        else:
                            self.logger.warning(
                                f"Column '{col}' (Oracle type: {otype}) is not derived in mapping. Defaulting to string."
                            )
                            df[col] = df[col].astype(str)
                except Exception as e:
                    self.logger.warning(f"Could not enforce schema: {e}")

            self.logger.info(f"DataFrame (schema enforced) with {len(df)} rows")
            return df, len(df)

        except Exception as e:
            self.logger.error(f"Error executing query: {sql}")
            raise self.error_handler.handle_exception(e)
        finally:
            if conn:
                self.connection_pool.return_connection(conn)