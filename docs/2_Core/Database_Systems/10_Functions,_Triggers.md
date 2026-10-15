## Functions

returns a single value

```mysql
create function deptCountFunc(deptName varchar(30)) returns integer
begin
	declare dCount integer;

	select count(*) into dCount
	from instructor
	where instructor.deptName = deptCountFunc.deptName;
	
	return dCount;
end
```

### Invokation

can be called within a query only

```mysql
select deptName
from instructor
where deptCountFunc("Physics") > 5;
```

### Table Function

function that returns a table

```mysql
create function instructorOf(deptName char(20)) returns table (
  id varchar(5),
  name varchar(20),
  deptName varchar(20),
  salary numeric (10, 2)
)
	return table(
  	select id, name, deptName, salary
    from instructor
    where instructor.deptName = instructorOf.deptName;
  );
```

```mysql
select name
from instructorOf("Physics");
```

## Procedure

is like a void function that returns nothing

```mysql
create procedure deptCountProc (in deptName varchar(20),
                                out dCount integer)
begin
	select count(*) into dCount
	from instructor
	where instructor.deptName = deptCountProc.deptName;
end
```

### Invokation

can be called anywhere

- within a query, or

- outside everything else

  ```mysql
  declare dCount integer;
  call deptCountProc("Physics", dCount);
  ```

## Loops

### `while`

```mysql
while <booleanExpression> do
	; statements
end while
```

### `repeat`

is like `do while` in CPP

```mysql
repeat
	; statements
until <booleanExpression>
end repeat
```

### `for`

Find the budget of all departments

```mysql
declare totalBudget integer default 0;
for
	i as
		select budget from department
	do
		set totalBudget = totalBudget + i.budget
end for
```

## Triggers

statement that is executed automatically as a side effect of a modication of the database

```mysql
show triggers;
```

### Referencing

- `referencing old row as orow` - updates and deletes
- `referencing new row as nrow` - updates and inserts

```mysql
create trigger setnullTrigger
before update of takes
referencing new row as nrow
for each row
when(nrow.grade = "")
begin atomic
  set nrow.grade = null
  set nrow.attendance = 0
end;
```

```mysql
create trigger creditsEarned
after update of takes on(grade)
referencing new row as nrow
referencing old row as orow
for each row
when nrow.grade != 'F' and nrow.grade is not null
 and (orow.grade = 'F' or orow.grade is null)
begin atomic
	update student
	set totCred = totCred + (
  	select credits
    from course
    where course.cid = nrow.cid
  )
  where student.id = nrow.id;
end;
```

`begin atomic` means update everywhere
