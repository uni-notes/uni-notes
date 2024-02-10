# FRED API

The following is my own way to interact with the API, which re-uses the same client for the entire session

```
# requirements
httpx[http2]
aiolimiter
```

```python
import numpy as np
import pandas as pd

import httpx
import asyncio
from aiolimiter import AsyncLimiter
```

```python
class Fred:
  def __init__(self, API_KEY):
    self.set_api_key(API_KEY)

    self.URL_BASE = 'https://api.stlouisfed.org/'

    self.params = dict(
        api_key = API_KEY,
        file_type = "json",
        offset = 0,
        limit = 1000,
        sort_order = "asc"
    )

    self.create_client()

    self.create_limiter()

    self.series_data_id_list = []

  async def __del__(self):
      await self.delete_client()

  def set_api_key(self, API_KEY):
    self.API_KEY = API_KEY

  def create_client(self):
    if not hasattr(self, "client") or self.client.is_closed:
      self.client = httpx.AsyncClient(http2=True)

  async def delete_client(self):
    print("Deleting client")
    if hasattr(self, "client") and not self.client.is_closed:
      await self.client.aclose()

  def create_limiter(self):
    self.limiter = AsyncLimiter(
      10, # asynchronous requests
      1   # delay in s
    )


  async def get_series_single(
      self,
      series_id,
      start_date = None,
      end_date = None,
      **kwargs
  ):
    try:
      response_json = await self.get_data(
          'fred/series/observations',
          series_id = series_id,
          observation_start = start_date,
          observation_end = end_date,
          **kwargs
      )

      response_data = response_json["observations"]

      if len(response_data) == 0:
        return np.nan # None

      df = pd.DataFrame(response_data)[["date", "value"]]
      df["date"] = pd.to_datetime(
          df["date"],
          format = "%Y-%m-%d"
      )

      series = pd.to_numeric(
          df
          ["value"]
          .replace(
              ".",
              np.nan,
              regex=False
          )
      )

      series.index = df["date"]
      series.index.name = "Date"

      return series

    except Exception as e:
      print(series_id + ": " + str(e))

      return np.nan # None

  async def get_data(
      self,
      ENDPOINT,
      **kwargs
  ):
    params = self.params
    params.update(
        **kwargs
    )

    async with self.limiter:
      response = await self.client.get(
          self.URL_BASE + ENDPOINT,
          params=self.params
      )

    return response.json()


  async def get_series_list(
      self,
      tags = [],
      **kwargs
  ):

    try:
      response_json = await self.get_data(
          'fred/tags/series',
          tag_names = ";".join(tags),
          **kwargs
      )

      response_data = response_json["seriess"]

      if len(response_data) == 0:
        return np.nan # None

      return (
          pd.DataFrame(response_data)
      )

    except Exception as e:
      print(str(e))

      return (
          pd.DataFrame()
      )

  async def get_series_thread(self, series_id, **kwargs):
    series_data = await self.get_series_single(series_id, **kwargs)
    return (series_id, series_data)

  async def get_series(
      self,
      series_id_list,
      **kwargs
  ):

    tasks = []

    for series_id in series_id_list:
        tasks.append(asyncio.create_task(self.get_series_thread(series_id, **kwargs)))

    self.series_data_id_list = await asyncio.gather(*tasks)

    try:
      return pd.DataFrame({
          series_id :
          series_data for series_id, series_data in self.series_data_id_list
      })
    except Exception as e:
      print(str(e))
      return None
```

### Create Object

```python
fred = Fred(API_KEY)
```

### Filter Series

```python
series_list_df = await (
    fred
    .get_series_list(
      tags = ["india", "monthly"],
      limit = 2
    )
)
series_list_df
```

### Get series data

```python
series_list = list(series_list_df["id"])
series_data = await fred.get_series(
    series_id_list = series_list,
    limit = 1
)

series_data
```

