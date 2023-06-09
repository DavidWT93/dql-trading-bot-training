# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 12:02:51 2022

@author: Tillery
"""

import pymysql.cursors
from sqlalchemy import create_engine
import pandas as pd


class SQLHandling:
    def __init__(self, database, password, host='localhost', user="root",
                 port=3306):
        self.database = database
        self.host = host
        self.user = user
        self.password = password
        self.port = port
        self.db = pymysql.connect(host=self.host,
                                  user=self.user,
                                  password=self.password,
                                  port=self.port,
                                  database=self.database)
        self.cursor = self.db.cursor()

    def __del__(self):
        self.db.close()

    def get_column_names_from_table(self, tableName):
        sql = f"SHOW COLUMNS FROM {self.database}.{tableName};"
        self.cursor.execute(sql)
        cols = self.cursor.fetchall()
        columnsList = [cols[i][0] for i in range(len(cols))]
        return columnsList

    def get_table_as_df(self, tableName):
        sql = f"SELECT * FROM {tableName}"
        self.cursor.execute(sql)
        results = self.cursor.fetchall()
        results = list(results)
        columnNames = self.get_column_names_from_table(tableName)
        dataFrameOfTable = pd.DataFrame(results, columns=columnNames)
        return dataFrameOfTable

    def get_n_laste_entries_from_table_by_order_colum(self, tableName, columnToOrder, n):
        sql = f"SELECT * FROM {tableName} ORDER BY {columnToOrder} DESC LIMIT {n}"
        self.cursor.execute(sql)
        results = self.cursor.fetchall()
        results = list(results)
        columnNames = self.get_column_names_from_table(tableName)
        dataFrameOfTable = pd.DataFrame(results, columns=columnNames)
        return dataFrameOfTable

    def get_data_as_df_no_column_names(self, sql):
        self.cursor.execute(sql)
        result = self.cursor.fetchall()
        result = list(result)
        DF = pd.DataFrame(result)
        return DF

    def get_sql_query_as_list(self, sql):
        self.cursor.execute(sql)
        result = self.cursor.fetchall()
        result = list(result)
        return result

    def get_sql_query_as_df(self, sql):
        """
        with this funktion you can use a custom query and get the resulting
        output as a Dataframe

        Parameters
        ----------
        sql : custom sql query

        Returns
        -------

        df : resulting output as dataframe
        """
        self.cursor.execute(sql)
        result = self.cursor.fetchall()
        col_names = [i[0] for i in self.cursor.description]
        result = list(result)
        df = pd.DataFrame(result, columns=col_names)
        return df

    def update_sql_query(self, sql):
        print("SQL command:", sql)
        try:
            self.cursor.execute(sql)
            self.db.commit()
        except Exception as ex:
            print(ex)

    def create_table_of_df(self, dataFrame, tableName):
        """
        Parameters
        ----------
        dataFrame : TYPE
            DESCRIPTION.
        tableName : TYPE
            hier muss man beachten das der tabellen Name noch nicht existiert

        Returns
        -------
         None.

        mit der Methode kann man eine ganze DF in eine SQL Datenbank laden,
        oder eine DF in eine bestehende Tabelle laden

        """
        sqlEngine = create_engine(f'mysql+pymysql://{self.user}:{self.password}@{self.host}/{self.database}',
                                  pool_recycle=3600)
        dbConnection = sqlEngine.connect()

        try:
            frame = dataFrame.to_sql(tableName, dbConnection, if_exists='append')
        except ValueError as ve:
            print(ve)
        except Exception as ex:
            print(ex)
        finally:
            dbConnection.close()

    def create_table_of_df_set_index(self, dataFrame, tableName, indexColumn, uniqueId=False):
        """
        Parameters
        ----------
        uniqueId
        indexColumn
        dataFrame : TYPE
            DESCRIPTION.
        tableName: TYPE
            hier muss man beachten, das der tabellen Name noch nicht existiert

        Returns
        -------
         None.

        mit der Methode kann man eine ganze DF in eine SQL Datenbank laden,
        oder eine DF in eine bestehende Tabelle laden

        """
        sqlEngine = create_engine(f'mysql+pymysql://{self.user}:{self.password}@{self.host}/{self.database}',
                                  pool_recycle=3600)
        dbConnection = sqlEngine.connect()

        try:
            if uniqueId:
                dataFrame.to_sql(tableName, dbConnection, if_exists='append',
                                 index=False)
                sql = f"ALTER TABLE {tableName} ADD PRIMARY KEY ({indexColumn}(100))"
                self.update_sql_query(sql)
            else:
                dataFrame.to_sql(tableName, dbConnection, if_exists='append',
                                 index=False, index_label=indexColumn)

        except ValueError as ve:
            print(ve)
        except Exception as ex:
            print(ex)
        finally:
            dbConnection.close()

    def load_csv_file_to_db(self, csvFile, tableName, printDF=False):
        DF = pd.read_csv(csvFile, sep=";",
                         encoding="ISO-8859-1")
        if printDF: print(DF)
        self.create_table_of_df(DF, tableName)



