## Join

1st table’s order is always followed.

|                       | Outer                          | Inner                                                        | Natural                                       |
| --------------------- | ------------------------------ | ------------------------------------------------------------ | --------------------------------------------- |
| working               | uses `null` for missing values | selects all rows from both tables as long as there is a match between the columns. | only common tuples with only the left table   |
|                       |                                | Returns records that have matching values in both tables     |                                               |
| condition             |                                |                                                              | a column name in both the tables must be same |
| common table repeated | ✅                              | ✅                                                            | ❌                                             |

```mysql
select *
from t1 left outer join t2 on t1.roll = t2.rollNum;

select *
from t1 inner join t2 on t1.roll = t2.rollNum;

select *
from t1 natural join t2;
```

### Outer Join

| Left Outer Join                                              | Right Outer Join                                             | Full outer Join                                              |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Left table’s tuples will occur once                          | Right table’s tuples will occur once                         |                                                              |
| Returns all records from the left table, and the matched records from the right table | Returns all records from the right table, and the matched records from the left table | Returns all records when there is a match in either left or right table |
