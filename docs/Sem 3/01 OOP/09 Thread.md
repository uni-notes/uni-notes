## Threads

independent subprocess

Multi-threading allows for multiple subprocesses to occur for perform a task

## Processor

has 3 stages to perform tasks

$\fbox F \fbox D \fbox E$

- Fetch instruction
- Decode instruction
- Execute instruction

## Life Cycle of Thread

``` mermaid
flowchart LR
new & blocked & waiting --> runnable
runnable --> blocked & waiting & terminate
```

- new (born) state
    - `start` is called implicitly(on it’s own)
- Blocked state
    - paused / waiting for I/O or notification
    - short duration
- Waiting state
    - processor is busy
    - when finally going to runnable, `notify() / notifyall()` method is called
- runnable state
    - highest-priority thread enters
- terminate (Dead) state
    - thread has been processed
- sleeping state
    - `sleep(t)` is called
    - $t$ is in ms
    - long duration
    - exits this state when sleep timer has expired

## Priorities

- `Thread.MIN_PRIORITY` - 1
- `Thread.NORM_PRIORITY` - 5 (default)
- `Thread.MAX_PRIORITY` - 10

New threads inherit the priority of the thread that created it

## Timeslicing

Round robin fashion

The initial run takes place based on priority ensuring that each task gets run for 4s. Then, purely based on priority, tasks are run

In the following example, let’s say that priority is $T_2 > T_4 > T_3 > T_1$,

$$
\color{green}
\underbrace{
\underset{4s}{\fbox{$T_2$}}
\underset{4s}{\fbox{$T_4$}}
\underset{4s}{\fbox{$T_3$}}
\underset{4s}{\fbox{$T_1$}}
}_\text{initial run}
\color{orange}
\underbrace{
\underset{4s}{\fbox{$T_2$}}
\underset{4s}{\fbox{$T_2$}}
\underset{4s}{\fbox{$T_2$}}
\underset{4s}{\fbox{$T_4$}}
\underset{4s}{\fbox{$T_3$}}
\underset{4s}{\fbox{$T_3$}}
\underset{4s}{\fbox{$T_1$}}
}_\text{purely based on priority}
$$

## Implementation

Both methods are pretty much identical

- implementing `Runnable` (better)
- or, extending `Thread`
    - not recommended
    - cuz then we can’t inherit any other class (java doesn’t support class multiple inheritance)

### Implement `Runnable`

``` java
class MyThread implements Runnable 
{
  public void run() {
	  // logic    
	}
}

public class Tester
{
	Thread t1 = new Thread(new MyThread() );
	Thread t2 = new Thread(new MyThread() );
  
  // or
	Runnable r = new MyThread();
  Thread t = new Thread(r);
}
```

### Extending `Thread`

```java
class MyThread extends Thread
{
  public void run()
  {
    // logic
  }
}

class Tester
{
  public static void main()
  {
		MyThread t = new MyThread();
  }
}
```

## Inter Thread Communication

proper coordination/communication between thread helps take care of deadlock situation

``` java
MyThread t1 = new MyThread();

t1.start();
t1.sleep(4000); // ms
t1.join();
t1.suspend(); // stop it indefinitely, unless resumed
t1.resume();

t1.wait();
t1.notify();
t1.notifyAll();
```

synchronized ensures that only thread runs at a time

``` java
public static class PC
{
  public void produce() throws InterruptedException
  {
    synchronized(this)
    {

    }
  }

  // or

  public synchronized void produce() throws InterruptedException
  {
    System.out.println("Producer Thread Running");
    wait(); //wait tills another does notify()
    System.out.println("");
  }

  public synchronized void consume() throws InterruptedException
  {
    Thread.sleep(1000);
    Scanner inp = new Scanner(System.in);
		
		synchronized(this)
    {
      System.out.println("Consumer Thread Running");
      inp.nextLine();
      System.out.println("Return Key Pressed");
      
      notify();
      
      Thread.sleep(2000);
    }

  }
}

class Tester
{
  public static void main()
  {
    final PC p = new PC();
		
    // create a thread object that calls pc.produce()
    Thread t1 = new Thread(new Runnable())
    {
      @Override
			public void run()
      {
        try {
          pc.produce();
        } catch(InterruptedException e) {
          e.printStackTrace();
        }
      }
    }; // anonymous class ends with ;
    
    Thread t2 = new Thread(new Runnable())
    {
      @Override
			public void run()
      {
        try {
          pc.consume();
        } catch(InterruptedException e) {
          e.printStackTrace();
        }
      }
    }; // anonymous class ends with ;
    
    t1.start();
    t2.start();
    
    // when t1 finishes, t2 starts
    // then t1 finishes, t1 starts
    t1.join();
    t2.join();
  }
}
```

## `Thread` methods

```java
public void start(); 
public void run(); // contains logic of the thread
// logic should always be inside try block
// and there should also be a catch block with InterruptedException

join();
wait(); // sends thread to wait state
resume(); // takes thread out of block state
suspend(); // sends thread to block state
notify();
notifyAll();

public final boolean isAlive(); // check if alive 
public static void sleep(long millisec); // send thread to block state for a while
public final void setDaemon(boolean on); // set this thread as a daemon thread
public void interrupt(); // not sure
public static void yield(); // give the runtime to other threads with the same priority

public final void setName(String name);
public final void setPriority(int priority); // 1 <= p <= 10
```

## Daemon Thread

is a low priority thread that runs in the background, to perform tasks such as garbage collection
