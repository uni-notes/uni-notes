# Custom IoT Server-Side Solution

This section will cover how to make our own IoT platform, using AWS EC2 (Amazon Web Services - Elastic Compute Cloud)

Virtual Private Server

Warning: Be careful about billings; only create one server and only use whatever services you are sure about

## Alternatives to AWS

- Azure
- Google Cloud
- Digital Ocean
- Linode

## Setup

- EC2 dashboard
- Change datacenter location to preferred location
- Launch Instance



1. Choose AMI (Amazon Machine Image)
   - Operating system
   - CPU architecture
2. Choose Instance Type
   - hardware configurations
3. Choose instance details: You may skip this
4. Add storage
5. Add tags: you may skip this
6. Configure security group
   1. Add rule: SSH
   2. Add rule: HTTP
7. Review instance launch
8. Create a new key pair

### Operating System

- Windows
- Linux (preferred)
  - FOSS
  - Secure
  - Fast

## Installations

1. Select instance
2. Click `Connect`
3. Get public IP address
4. Click `Connect`
5. Paste the below into the terminal Terminal

| Purpose                  | Option                           |
| ------------------------ | -------------------------------- |
| Web Server               | Apache Server/<br />NGINX Server |
| API Programming Language | PHP/<br />Python                 |
| Database                 | MySQL                            |
| Database Management Tool | PHPMyAdmin                       |

```bash
# update all packages
sudo apt-get update

# install apache server
sudo apt-get install apache2

# start apache server
sudo service apache2 start
 
# install php
sudo apt-get install php-dev libmcrypt-dev gcc make autoconf libc-dev pkg-config

# restart apache server
sudo service apache2 restart
 
# change directory
cd /etc/apache2/

# instal mysql server
sudo apt-get install mysql-server

# Setup MySQL security
sudo mysql_secure_installation
# Press N for validate password component
# Create password
# Press Y for remaining questions

# install phpmyadmin
sudo apt-get install phpmyadmin
# click yes for the questions

# link phpmyadmin to mysql
sudo ln -s /etc/phpmyadmin/apache.conf /etc/apache2/conf-available/phpmyadmin.conf
sudo a2enconf phpmyadmin.conf
sudo service apache2 reload
sudo systemctl restart apache2
sudo chmod -R 777 /var/www/html

# login to mysql
sudo mysql -uroot -p

# create admin user with password for phpmyadmin
CREATE USER 'admin'@'localhost' IDENTIFIED BY 'give_good_password';
GRANT ALL PRIVILEGES ON *.* TO 'admin'@'localhost';

# exit
exit
```

## phpmyadmin

Verify phpmyadmin

1. Go to `http://public_ip/phpmyadmin`
2. Put username and password from what was inputted for phpmyadmin in terminal

IDK

1. Create database
2. Create table
3. Create columns

## API

### `basic.php`

```php
<?php

$host	= "localhost";
$user	= "admin";
$pass	= "password";
$db		= "iot";
  
// connect to mysql
$con = mysqli_connect(
	$host,
  $user,
  $pass,
  $db
);

if ($con) {
  echo "Connection successful!";
} else {
  echo "Connection failed!";
}

?>
```

### `insert.php`

`http://.../insert.php?query_param=100&query_param_2=200`

```php
<?php
data_default_timezone_set("Asia/Kolkata");

$host	= "localhost";
$user	= "admin";
$pass	= "password";
$db		= "iot";
  
// connect to mysql
$con = mysqli_connect(
	$host,
  $user,
  $pass,
  $db
);

if ($con) {
  
  $query_param = $_GET["query_param"];
  
  if ($query_param) {
    // to ensure empty values not inserted
    
    $date = date("Y-m-d");
    $time = date("H:i:s");

    $query = "
    insert into table_name
    (date, time, data)
    values(
    '$date', '$time', $data
    );  
    ";

    if (mysqli.query($con, $sql)) {
      echo "Data inserted!";
    } else {
      echo "Insert failed!";
    }
  } else {
    echo "Missing query parameter (s)";
  }
      
} else {
  echo "Connection failed!";
}

?>
```

### `get_readings.php`

`http://.../get_readings.php`

```php+HTML
<html>
<head>
	<meta http-equiv="refresh" content="5">
</head>
<body>
<table>
<tbody>
  
<?php
$host	= "localhost";
$user	= "admin";
$pass	= "password";
$db		= "iot";
  
// connect to mysql
$con = mysqli_connect(
	$host,
  $user,
  $pass,
  $db
);

if ($con) {
  
  $query = "
  select * from table_name
  order by id desc
  limit 100
  ";
  
  $query_result = mysqli.query($con, $sql)

  if ($query_result) {
    
    while (
      $row = mysqli_fetch_array($query_result)
    ){
      // print_r($row);
      echo "
      <tr>
      <td>$row['data']</td>
      </tr>
      "
    }
    
  } else {
    echo "Query failed!";
  }
      
} else {
  echo "Connection failed!";
}

?>
</tbody>
</table>
</body>
</html>
```

### `set_status.php`

```php+HTML
<html>
<title>Cloud Server Controlled LED</title>
<body>
<center>
<h2 style='margin-top:50px;color:#123456;'>Cloud Server Controlled LED</h2>
<a href='?status=on'>
	<button style='background-color:green;
	  font-size:20px;color:white;margin:10px;padding:5px;'>
	  <b>LED ON</b>
	</button>
</a>
<a href='?status=off'>
	<button style='background-color:red;font-size:20px;color:white;
	margin:10px;padding:5px;'>
	<b>LED OFF</b>
	</button>
</a>
</center>
</body>
</html>
 
<?php
if(isset($_GET['status'])){
	date_default_timezone_set("Asia/Kolkata");
	
  $host = "localhost";
	$user = "iot_user";
	$pass = "iot@1122";
	$db = "iot";
	
  $con = mysqli_connect($host,$user,$pass,$db);
	
  $date = date("d-m-Y");    // 06-01-2022
	$time = date("H:i:s");
	$status = $_GET['status'];
	$query = "
	insert into led
	(date,time,status)
	values('$date','$time','$status')
	";
	
  mysqli_query($con, $query);
}

?>
```

### `get_status.php`

```php+HTML
<?php

$host = "localhost";
$user = "iot_user";
$pass = "iot@1122";
$db = "iot";

$con = mysqli_connect($host,$user,$pass,$db);

$query = "
select *
from table_name
order by id desc
limit 1
";

$query_result = mysqli_query($con, $query);

while(
  $row = mysqli_fetch_array($result)
) {
  echo $status;
  
  break; // only one row any ways
}

?>
```

## Uploading API files

Using: Filezilla

- General
  - File > Site Manager
  - New Site
    - Protocol: SFTP
    - Host: ip of the remote machine
    - Logon type: Key file
- Advanced
  - Default remote directory: `/var/www/html`
- Upload php files