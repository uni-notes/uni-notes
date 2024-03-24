# Forms

- Gallery
  Onselect: Set(SelectedItem, ThisItem)
- Form
  Datasource: Live Source
- Item: SelectedItem

## DisplayMode

|      |      |
| ---- | ---- |
| New  |      |
| Edit |      |
| View |      |

## IDK

For date inputs, Set `IsEditable` to `false`

`Onsuccess`

```
Notify("Success");
Back();
ResetForm(Form_Name);
```

`Onfailure`

```
Notify("Failure", Form_Name.Error, Form_Name.ErrorKind)
```

## Dropdown for text column

```
// Default_selected_items 
[ LookUp(StorageLocationbyRegion, StorageLocation = Parent.Default) ]
```

## Copy Form

```
If(
  use_previous_data,
  First(DropColumns(Table(Create_Item.LastSubmit), "ID")),
  Blank()
)
```

