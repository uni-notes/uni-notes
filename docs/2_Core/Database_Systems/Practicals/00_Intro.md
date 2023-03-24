## Creating Connection

1. Open Workbench
2. Click +
3. Enter any connection name (i put my uni ID 2020A7PS0198U)
4. Enter hostname as 172.16.100.8
5. Enter username as collegeid (like 2020A7PS0198U)

## JDBC

## Creation

1. File > New Project
1. Java with Ant
1. Next > Finish

## Code

```java
package jdbc;
import java.sql.*;

public class JavaApplication7 {
	public static void main(String[] args) {
		query("salesman");
		query("instructor");
		query("takes");
    
    query("salesman", "salesman_id > 5003");
	}
  public static void query(String table)
  {
    query(table, "");
  }   
	public static void query(String table, String where)
	{
		try
		{
			String url = "jdbc:mysql://172.16.100.8/20200198db",
				user = "2020A7PS0198U",
				password = "a",
				query = "select * from " + table;
                        if(where!="")
                            query += " where " + where;
                        
			System.out.println(query + " 😊");
			
			Class.forName("com.mysql.cj.jdbc.Driver");
			Connection con = DriverManager.getConnection(url, user, password);
			Statement stmt = con.createStatement();
			ResultSet rs = stmt.executeQuery(query); 
			while(rs.next())
			{
				String col1 = rs.getString(1),
					col2 = rs.getString(2);
				System.out.println(col1 + " " + col2);
			}

			rs.close();
			stmt.close();
			con.close();
			
			System.out.println("");
		}
		catch(Exception e)
		{
			System.out.println("Something Happened 🤣");
		}
	}   
}
```

## Output

```
select * from salesman 😊
5001 James Hoog
5002 Nail Knite
5003 Lauson Hen
5005 Pit Alex
5006 Mc Lyon
5007 Paul Adam

select * from instructor 😊
102 ABC
103 DEF
104 GHI

select * from takes 😊
198 CS F111
199 Bio F111
200 Mech F111
201 111

select * from salesman where salesman_id > 5003 😊
5005 Pit Alex
5006 Mc Lyon
5007 Paul Adam
```

## GUI

## Steps

## Code

```java
private void jButton1ActionPerformed(java.awt.event.ActionEvent evt) {                                         
  String salesman_id = jTextField1.getText(),
  name = jTextField2.getText(),
  city = jTextField3.getText(),
  commission = jTextField4.getText();

  String table = "salesman",
  values = "'" + salesman_id + "', '" + name + "', '" + city + "', " + commission + "'";

  insertQuery(table, values);
}                                        

public void insertQuery(String table, String values)
{
  try
  {
    String url = "jdbc:mysql://172.16.100.8/20200198db",
    user = "2020A7PS0198U",
    password = "a",
    query = "insert into " + table + " values("  + values + ")";

    System.out.println(query);

    Class.forName("com.mysql.cj.jdbc.Driver");
    Connection con = DriverManager.getConnection(url, user, password);
    Statement stmt = con.createStatement();
    stmt.executeUpdate(query);

    stmt.close();
    con.close();

    //                        JOptionPane.showMessageDialog(this, query);
  }
  catch(Exception e)
  {
    System.out.println("Something Happened 🤣");
  }
}   
```
