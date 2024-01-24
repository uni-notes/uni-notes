## `Patch`

```javascript
Patch(
    List_Name,
    {
        Col1: "foo"
    }
)
```

## Error Handling

```
Set(errors, Errors(List_Name));
If(
  IsEmpty(errors),
  Notify("Success", Notification.Success),
  Notify(First(error).Message, Notification.Error)
);
```

## 

## Advanced

For creating record, use

```
Patch(
  Data_Source,
  Table({
    Title:"Num1",
      number:1
  })
)
```

instead of

```
Patch(
	dummyData,
  Defaults(dummyData),
  {Title:"Num1",number:1}
)
```

For updating record, use

```
Patch(dummyData,{ID:1},{Title:"Num1",number:1})
```

instead of

```
Patch(dummyData,
  LookUp(dummyData,ID=1),
  {Title:"Num1",number:1}
)
```

Batch patching

```javascript
Patch(
  Data_Source_Name,
  ShowColumns(Collection_Name, "ID", "FullName", "Status")
)

// idk
ForAll(
    Add_Users_Input.SelectedItems As user_to_add,
    Collect(
        users_to_add,
        Table({
        Name: user_to_add,
        Country_Code: LookUp(Choices([@Users].Country_Code), Value=country_code),
        Superuser: false
      })
    )
);
IfError(
    Patch(
        'Inventory App Multiple Region Users',
        users_to_add
    );
    Notify(
        "Successfully added users",
        NotificationType.Success
    );
    Reset(Add_Users_Input);
    ,
    Notify(
        "Addition of users failed",
        NotificationType.Error
    );

);
Clear(users_to_add);
```

instead of

```javascript
ForAll(
  colUpdateEmployees,
  Patch(
      Employees,
      LookUp(Employees, ID=colUpdateEmployees[@ID]),
      {
        FullName: colUpdateEmployees[@FullName],
        Active: colUpdateEmployees[@Active]
      }
    )
)
```

## People Column

```js
Patch(
  data_source,
  {
	  Person_Column: {
        '@odata.type': "#Microsoft.Azure.Connectors.SharePoint.SPListExpandedUser",
        Department: "",
        Claims: "i:0#.f|membership|" & User().Email,
        DisplayName: Office365Users.UserProfileV2(User().Email).displayName,
        Email: User().Email,
        JobTitle: "",
        Picture: ""
      }
  }
)
```

