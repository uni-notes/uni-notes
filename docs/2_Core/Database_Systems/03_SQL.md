## SQL

is a non-procedural language

## Database

keywords are not case-sensitive

```mysql
create database dbName;
show databases;
use dbName;
```

## DDL

Data Definition Language

work with structure

```mysql
## Create Table
create table tableName(
  col1 dataType(size),
  col2 dataType(size),
  col3 dataType(size)
);

drop table student;
```

### Constraints

1. Primary Key
2. Foreign key
3. Cascading
4. not null

```mysql
create table tableName(
  col1 dataType(size),
  col2 dataType(size),
  col3 dataType(size),
  col4 dataType(size),

  primary key(col1, col2),
  foreign key(col3),
  not null(col4),
  on delete cascade(col1, col2, col3, col4)
);

ALTER TABLE department ADD PRIMARY KEY (dept_name);

alter table orders modify column purch_amt float(10,5);
```

## DML

Data Manipulation Language

work with entries

```mysql
## Display properties of table
describe Students;

## Insert
insert into Students values(1, "Thahir", "Database Systems");

## Insert Multiple
insert into Students values
(1, "Thahir", "Database Systems"),
(2, "Blah", null),
(3, "Blah", null);
```

### `deletion`

```mysql
delete
from instructor;

truncate instructor;

delete
from instructor
where deptName = "Finance";

delete
from instructor
where deptName in (
	select deptName
  from departments
  where building = "Watson"
);

delete
from instructor
where salary < (
	select avg(salary)
  from instructor
);
```

### `update`

```mysql
update instructor
set salary = salary * 1.03
where salary > 100000;
	
update instructor
set salary = case when salary <=10000
	then salary * 1.05
	else salary * 1.03;
		
update student
set totCreds = (
	select sum(credits)
  from takes
  where takes.cid = (
  	select course.cid
    from course
  )
  	and student.id = takes.id
  	and takes.grade != 'F'
  	and takes.grade is not null
);

update student
set totCreds set null somethign is here
```

