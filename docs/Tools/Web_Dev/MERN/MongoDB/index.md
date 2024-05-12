# MongoDB

MongoDB does not require explicit creation. If you try to access something that doesn’t exist, MongoDB will create it for you.

## Install 

1. Install `MongoDB Community`
2. Install `mongosh` (shell)

## Vocabulary

| Relational  | MongoDB    |
| ----------- | ---------- |
| Database    | Database   |
| Table       | Collection |
| Column      | Key        |
| Row         | Document   |
| Index       | Index      |
| Join        | $lookup    |
| Foreign Key | Reference  |

## Data Format

BSON (Binary JSON): very similar to json

## Mongosh

```js
mongosh // enter
exit // exit
```

## DDL

```js
show dbs
use appdb
show collections
db.dropDatabase()
```

## DML

### Create

```js
db.users.insertOne({
	name: "Ahmed"
})
db.users.insertMany([
  {
    name: "Ahmed"
  },
    {
    name: "Thahir"
  }
])
```

### Read

```js
db.users.findOne()
db.users.find() // equiv to select *
```

#### Find functions

```js
.sort({
	name:1, // -1
	age: 1
})
.skip(5)
.limit(10)
```

#### Count

```js
db.users.countDocuments({
  age: 10
})
```

### Update

`$set`

```js
db.users.updateOne(
  {
  	age: 26
	},
  {
    $set: {age: 27}
  }
)
db.users.updateMany(
  {
  	age: 26
	},
  {
    $set: {age: 27}
  }
)
```

```js
$set
$inc
$rename: {name: "Ahmed"}
$unset: {name: ""} // removes key; doesn't set to null
$push: {hobbies: "Swimming"} // adding to array key
$pull: {hobbies: "Swimming"} // remove from array key
```

### Replace

```js
db.users.replaceOne(
  {
  	age: 26
	},
  {
    name: "Thahir"
  }
)
db.users.replace(
  {
  	age: 26
	},
  {
    name: "Thahir"
  }
)
```

### Delete

```js
db.users.deleteOne(
  {
    name: "Thahir"
  }
)
db.users.deleteMany(
  {
    name: "Thahir"
  }
)
```

### Filtering

```js
// filter Thahir and return only name and age
db.users.find(
  {
    name: "Thahir"
  },
  {
    name: 1,
    age:1,
    _id: 0
  }
)
```

#### Complex Filters

```js
db.users.find(
  {
    first_name: {$eq: "Ahmed"},
    age: {$gte: 50},
  },
  {
    name: 1,
    age:1,
    _id: 0
  }
)
```

```js
{$eq: "Thahir"}
{$ne: "Thahir"}
{$gte: 10}
{$in: [
  "Ahmed",
  "Thahir",
  5
]}
{$nin: [
  "Ahmed",
  "Thahir",
  5
]}
{$exists: true} // only checks if key exists; hence includes documents with null
{$exists: false}
```

#### Filter Operations

```js
// not
{
	not: [
    {filter_1: "enset"}
  ]
}

// and
{
	filter_1: "enset",
  filter_2: "enset"
}

{
	$and: [
    {filter_1: "enset"},
	  {filter_2: "enset"}
  ]
}

// or
{
	$or: [
    {filter_1: "enset"},
	  {filter_2: "enset"}
  ]
}
```

#### Comparing keys

```js
db.users.find({
  $expr: {
    $gt: ["$debt", "$balance"]
  }
})
```

#### Nested keys

```js
db.users.find({
  "address.street": "Testing"
})
```

### 

## References

- [ ] [Web Dev Simplified | MongoDB Crash Course](https://www.youtube.com/watch?v=ofme2o29ngU)
