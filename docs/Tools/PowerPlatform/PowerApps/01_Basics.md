# Basics

## Variables

|         |                             |                                    |
| ------- | --------------------------- | ---------------------------------- |
| Global  | Entire app                  | `Set(variable, true);`             |
| Context | Local variables to a screen | `UpdateContext({variable: true});` |

## Basic Functions

```javascript
Navigate('Screen Name');
Concurrent(
	action1,
	action2
);
Select('Button Name'); // always should be at end of code block
```

## Timer

It does not work on code view

## Gallery

```js
Select(
  Tab,
  current_tab_selected,
  Tab_Button
);
```

