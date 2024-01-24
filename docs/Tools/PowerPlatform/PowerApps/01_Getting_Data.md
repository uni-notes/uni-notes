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

- Delegable
  - `Filter()``
  - ``Lookup()`
- Non-Delegable
  - `AddColumns`, `ShowColumns`, `DropColumns`, `RemoveColumns`

Magical blue squigglies only come out on operators, not on functions. So you would not see that showing a warning.
