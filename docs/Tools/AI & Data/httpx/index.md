# httpx

```python
# requirements
httpx[http2]
aiolimiter
```

```python
async def get_data(client, limiter, series):
  async with limiter:
    response = await client.get(
        URL_BASE + ENDPOINT + series,
        params = params
    )

  return response.json()

  
async with httpx.AsyncClient(http2=True) as client:
  limiter = AsyncLimiter(
    10, 	# asynchronous requests
    1 		# delay in s
  )
  
  tasks = []

  for series in series_list:
      tasks.append(asyncio.create_task(
          get_data(client, limiter, series)
      ))

  data = await asyncio.gather(*tasks)
```

