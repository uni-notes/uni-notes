# DuckDB

```python
import duckdb as db
import pandas as pd
```

```python
class DB():
  def __init__(self, db_name = "file.db"):
    self.db_name = db_name

  def execute(self, query):
    with duckdb.connect(self.db_name) as con:
      try:
          return con.sql(query).df()
      except:
        return False

  def create(self, table_name = "Series"):
    if table_name == "Series":
      query = f"""
      create or replace
      table {table_name}
      (
        Date Datetime,
        Variable String,
        Value Float,

        PRIMARY KEY (Date, Variable)
      )
      """
    elif table_name == "Variables":
      query = f"""
      create or replace
      table {table_name}
      (
        id String,
        realtime_start Datetime,
        realtime_end  Datetime,
        title String,
        observation_start Datetime,
        observation_end	Datetime,
        frequency String,
        frequency_short String,
        units String,
        units_short String,
        seasonal_adjustment String,
        seasonal_adjustment_short String,
        last_updated Datetime,
        popularity int,
        group_popularity int,
        notes String,

        PRIMARY KEY (id)
      )
      """
    
    return self.execute(query)
  
  def read(self, table_name, pivot=True):
    query = f"""
    select *
    from {table_name}
    """

    query_result = self.execute(query)

    if query_result is False:
      return False

    df = query_result
    
    if pivot:
      if table_name == "Series":
        df = df.pivot(
            index = "Date",
            columns = "Variable",
            values = "Value"
        )
      elif table_name == "Variables":
        pass

    return df
  
  def upsert(self, table_name, df, melt=True):
    if table_name == "Series":
      if melt:
        df = (
            df
            .copy()
            .reset_index()
            .melt(
                id_vars = "Date",
                var_name="Variable",
                value_name = "Value"
            )
            .dropna()
        )
      query = f"""
      INSERT INTO {table_name}
      select * from {var(df)}
      ON CONFLICT (Date, Variable)
      do update set Value = EXCLUDED.Value 
      """
    elif table_name == "Variables":
      if melt:
        pass
      
      query = f"""
      INSERT OR REPLACE INTO {table_name}
      select * from {var(df)}
      """
    
    return self.execute(query)
```

```python
economic_db = DB("economic.db")

# economic_db.create("Variables")
# economic_db.upsert("Variables", series_list_df)
economic_db.read("Variables")

# economic_db.create("Series")
# economic_db.upsert("Series", series_data, melt=True)
economic_db.read("Series", pivot=True)
```