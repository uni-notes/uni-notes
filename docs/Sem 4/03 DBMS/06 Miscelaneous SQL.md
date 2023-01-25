## Referential Integrity

Ensuring that the tuples in foreign table are in the main tables as well.

### Cascading

## Check Clause

### Complex

However, subqueries in check clause is not supported. Hence, triggers are preferred

## Indexing

speeds up querying, by using the indexes instead of looking at all records

```mysql
create table student(
	id int(4)
  primary key(id);
);

create index idIndex on student(id);

select id
from student
where age > 10;
```

## User-Defined Types

```mysql
create type dollars as numeric(12, 2) final

create table department(
	budget dollars;
)
```

## Domains

```mysql
create domain name char(20) not null
```

## Large-Object Types

Photos, videos, files are stored as a **large object**.

| blob                                          | clob                   |
| --------------------------------------------- | ---------------------- |
| binary large object                           | character large object |
| large collection of uninterpreted binary data | character dat          |
|                                               |                        |

(some point)

## Authorization

- Read (select)
- References (allow to create foreign key)
- Insert
- Update
- Delete
- Index
- Resources
- Alteration
- Drop

### Granting

```mysql
grant <privilegeList> on tableName/viewName to <userList>

grant select on instructor to user1, user2, user3
grant all privileges instructor to user1, user2, user3

create view geo_view as (select * from instructor where deptName = "Geology");
grant select on geo_view to geo_staff;
```

### Revoking

```mysql
revoke <privilegeList> on tableName/viewName from <userList>

revoke select on instructor from user1, user2, user3
revoke all privileges instructor from user1, user2, user3
```

## Roles

```mysql
create role roleName;
grant roleName to userName;

create role instructor;
grant instructor to Sapna;
```

Priveleges can be granded/revoked from roles as well

```mysql
grant select on takes to instructor;
```

### Chain of roles

```mysql
create role dean;
grant instructor to dean;
grant dean to Kumar;
```