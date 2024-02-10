# Miscellaneous

Need help with classifying these concepts

## Onvisible

```js
Set(downtime_status, false);
If(
    downtime_status,
    Notify("Due to downtime associated with global deployment, please use the app after an hour.", NotificationType.Error),
    false
);

Concurrent(
    Set(
        primary_font,
        "Segoe UI"
    ),
    Set(
        grey_font_color,
        RGBA(
            0,
            0,
            0,
            0.6
        )
    )
);

Set(
    user_details,
    Office365Users.MyProfile()
);
```

## Filtering

### Conditional Filtering

```js
Filter(
	collection,
  If(
    Field_1_Filter_Case_1,
    Field_2_Filter_Case_2
    )
)
// instead of 
If(
  Filter_1(),
  Filter_2()
)
```

### Fuzzy Search

Fuzzy search only works for collections! Use the performance tips to optimize the loading of data from datasource into collections.

```javascript
// <- or operation of search strings
true in ForAll()

// <- and operation of search strings
Not(false in ForAll())
```

```javascript
Filter(
  collection_name,
  If(
    Not(false in ForAll( // <- and: match all substrings
      Split(
        Trim(Substitute(Search_Input.Text, " ", "")), // Trim(Search_Input.Text),
        " "
      ) As substring,
      true in ForAll( // <- or: match any column
        [
          Col1,
      		Col2,
      		Col3
        ] As column, // columns to search
        substring.Value in Substitute(column, " ", "")
    		// Use substring.Value.Value for choice columns
    )
    )),
    true,
    false
  )
)
```

## Alerts

```
Notify( "Wrong", NotificationType.Warning, 4000 )
// Message, Type, Timeout
```

| NotificationType                       | Purpose                                |
| -------------------------------------- | -------------------------------------- |
| NotificationType.Error                 | Displays the message as an error.      |
| NotificationType.Information (Default) | Displays the message as informational. |
| NotificationType.Success               | Displays the message as success.       |
| NotificationType.Warning               | Displays the message as a warning.     |

Set `App.ConfirmExit` to `true`

## Alternating Colors

https://devoworx.net/alternate-row-color-in-gallery-powerapps/#alternate-row-color-in-gallery    

```
With(
{
    Items:List_or_List_Name
},
ForAll(
    Sequence(CountRows(Items)),
    Patch(
        Last(
            FirstN(Items,Value)),
            {rowNumber: Value}
        )
    )
)
```

```
If(
  Mod(ThisItem.rowNumber,2) = 0,
  RGBA(240, 240, 240, 1),RGBA(255, 255, 255, 1)
)
```

## Embed PowerBI

- PowerBI > `Embed` > `Website/Portal`

- PowerApps > `Insert` > `Charts` > `PowerBI Tile`
  - Select it
  - `TileUrl` > Paste embed link


## IDK

- Use `Select()` for re-using code by calling a button

  - Mainly useful for onvisible to call the refresh button

- Default for multi-select columns

	- ```
    {Value: ThisItem.Purpose}
    ```

## Home page onvisible

```javascript
Set(downtime_status, false);
If(
    downtime_status,
    Notify("Due to downtime associated with global deployment, please use the app after an hour.", NotificationType.Error),
    false
);

Concurrent(
    Set(
      user_details,
      Office365Users.MyProfile()
  ),
    Set(
        primary_font,
        Font.'Segoe UI' // "Custom Font"
    ),
    Set(
        grey_font_color,
      RGBA(0,0,0,0.6)
    ),
    Set(
        app_name,
        "Thahir App"
    )
);
```

Onvisible only once

```
If(
  !loadapp,
  ClearCollect(
    collection,
    Filter(datasource,condition)
  );
  UpdateContext({loadapp: true}),
  Blank()
)
```

## App Name

```javascript
app_name & " | " & App.ActiveScreen.Name
// set app_name globally
```

## Get first names

Multi-person column

```javascript
// Ahmed
// Ahamed
// Mohammed

ForAll(
      users_all As user,
      First(
          Split(
              Last(
                  Split(
                      user.Name.DisplayName,
                      ", "
                  )
              ).Value,
              " "
          )
      ).Value
  )
```

From a List to a string

```js
// Ahmed; Ahamed; Mohammed

Concat(
    ForAll(
        users_all As user,
        First(
            Split(
                Last(
                    Split(
                        user.Name.DisplayName,
                        "; "
                    )
                ).Value,
                " "
            )
        ).Value
    ),
    Value,
    Char(10)
)
```

From a table to a string for a cell in a Gallery

```js
// Ahmed; Ahamed; Mohammed
Concat(
    ForAll(
        ThisItem.Owner As ItemOwner,
        First(
            Split(
                Last(
                    Split(
                        ItemOwner.DisplayName,
                        "; "
                    )
                ).Value,
                " "
            )
        ).Value
    ),
    Value,
    ", "
)
```

default selected items for a dropdown

```js
// Ahmed; Ahamed; Mohammed
Filter(
    AddColumns(
        ForAll(
            users_all,
            Name
        ),
        "FirstName",
        First(
            Split(
                Last(
                    Split(
                        DisplayName,
                        "; "
                    )
                ).Value,
                " "
            )
        ).Value
    ),
    Email in ForAll(
        ThisItem.Owner, // don't use Parent.Default, as it gives weird results
        Email
    )
)
```

## Distinct Values from Choice Column

Single-Select

```js
Distinct(
    ForAll(
        List,
        Column
    ),
    Value
)
```

Multi-Select

```js
Distinct(
    Ungroup(
        ForAll(
            ForAll(
                List,
                Column
            ),
            Value
        ),
        "Value"
    ),
    Value
)
```

## Union of 2 tables


```js
Ungroup(
  Table(
    {MyTables: TableA},
    {MyTables: TableB}
  ),
  "MyTables"
)
```

```js
Ungroup(
    ForAll(
        Conditions_Time_Points_Gallery.AllItems,
        {
            MyTables: ForAll(
                Samples_Count_Gallery.AllItems,
                Samples_Count_Input.Text
            )
        }
    ),
    "MyTables"
)
```


## Replicating database-style input

### Input boxes

Onselect

```
Select(Update Record Button);
```

### Update Record Button

```js
If(
    LookUp(
        collection_modified,
        ID = ThisItem.ID,
        true
    ),
    Blank(),
    Collect(
        collection_modified,
        ThisItem
    )
);
UpdateIf(
    collection_modified,
    ID = ThisItem.ID,
    {
        Comments: Comments_Input.Value
    }
);
```

### Submit Button

```js
Patch(
    Success_Criteria,
    ShowColumns(
      collection_modified,
      "ID",
      "Col1",
      "Col2"
    )
);
```

## Deep Linking

### `App.Onvisible`

```js
Set(
    param_item_id,
    Param("item_id")
);
If(
  !IsBlank(param_item_id),
	Set(
    selected_stability_code,
    param_item_id
  );
	Select(go_to_manage_item);
)
```

### `go_to_manage_item` button

Create a button to do the navigation, because PowerApp does not allow `Navigate()` in onvisible

```js
Navigate('Screen_to_Manage_Item');
```

### Generate Link

```
https://<applink>?param1=value1&param2=value2&param3=value3
```

For example,

```
https://<applink>?item_id=100
```
