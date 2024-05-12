## Get Data

### Single Row

```js
Lookup(
  List,
  Cond1 && Cond2
)
```

### Multiple Rows

```js
Filter(
  List,
  Cond1,
  Cond2
)
```

## Delegability

Only concerns online data sources, ==**not collections**==

- Delegable
  - `Filter()`
  - ``Lookup()`
- Non-Delegable
  - `AddColumns`, `ShowColumns`, `DropColumns`, `RemoveColumns`

Magical blue squigglies only come out on operators, not on functions. So you would not see that showing a warning.

## `In vs LookUp`

```python
my_list = [1, 2, 3, 4, 5, ..., 1000]
key = 2

# in
for i in range(len(my_list)):
  if my_list[i] == key:
    print("found")
    
# lookup
for i in range(len(my_list)):
  if my_list[i] == key:
    print("found")
    break
```

## IDK

If you are testing delegability, set the delegation limit as 1, and check if everything works
