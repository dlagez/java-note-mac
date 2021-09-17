# 一个Runnable开启了多个线程

java核心技术卷一第12章并发里面的问题
疑问：我只运行了一个Runnable()，为什么会创建这么多线程。
有个一个银行实体类：

```javascript
package concurrent.threads;

import java.util.Arrays;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

public class Bank {
    // 存储用户，就是金额
    private final double[] accounts;
    // 并发锁
    private Lock bankLocks;
    // 资金充足条件
    private Condition sufficientFunds;

    /**
     * constructs of bank
     * @param n  初始化账户的个数
     * @param initialBalance
     */
    public Bank(int n, double initialBalance) {
        accounts = new double[n];
        Arrays.fill(accounts, initialBalance);
        bankLocks = new ReentrantLock();
        sufficientFunds = bankLocks.newCondition();
    }

    /**
     * transfer money from one account to another
     * @param from
     * @param to
     * @param amount
     */
    public void transfer(int from, int to, double amount) throws InterruptedException {
        bankLocks.lock();
        try {
            if (accounts[from] < amount) {
                // 资金不足，进入等待状态，并释放锁。
                sufficientFunds.await();
            }
            System.out.print(Thread.currentThread());
            accounts[from] -= amount;
            System.out.printf(" %10.2f from %d to %d", amount, from, to);
            accounts[to] += amount;
            System.out.printf(" Total Balance: %10.2f%n", getTotalBalance());
            // 完成转账，通知其他等待状态的线程，解除线程等待的阻塞。
            sufficientFunds.signalAll();
        } finally {
            // 释放锁
            bankLocks.unlock();
        }
    }

    /**
     * get the sum of all account balances
     * @return
     */
    public double getTotalBalance() {
        bankLocks.lock();
        try {
            double sum = 0;
            for (double account : accounts) {
                sum += account;
            }
            return sum;
        } finally {
            bankLocks.unlock();
        }

    }


    /**
     * get the number of accounts in the bank
     * @return
     */
    public int size() {
        return accounts.length;
    }
}
```

使用多线程来进行转账：

```java
package concurrent.threads;

public class UnsynchBankTest {
    public static final int NACCOUNTS = 100;
    public static final double INITIAL_BALANCE = 1000;
    public static final double MAX_AMOUNT = 1000;
    public static final  int DELAY = 10;

    public static void main(String[] args) {
        Bank bank = new Bank(NACCOUNTS, INITIAL_BALANCE);
        for (int i = 0; i < NACCOUNTS; i++) {
            int fromAccount = i;
            Runnable r = () -> {
                try {
                    while (true) {
                        int toAccount = (int) (bank.size() * Math.random());
                        double amount = MAX_AMOUNT * Math.random();
                        bank.transfer(fromAccount, toAccount, amount);
                        Thread.sleep((long) (DELAY * Math.random()));
                    }
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            };
            Thread t = new Thread(r);
            t.start();
        }
    }
}

```

输出：

```bash
Thread[Thread-75,5,main]     990.57 from 75 to 23 Total Balance:  100000.00
Thread[Thread-42,5,main]     150.01 from 42 to 48 Total Balance:  100000.00
Thread[Thread-74,5,main]     468.70 from 74 to 78 Total Balance:  100000.00
Thread[Thread-28,5,main]     903.20 from 28 to 91 Total Balance:  100000.00
Thread[Thread-48,5,main]     458.76 from 48 to 71 Total Balance:  100000.00
Thread[Thread-57,5,main]     797.88 from 57 to 56 Total Balance:  100000.00
Thread[Thread-22,5,main]     406.46 from 22 to 39 Total Balance:  100000.00
Thread[Thread-32,5,main]     740.97 from 32 to 80 Total Balance:  100000.00
Thread[Thread-82,5,main]     250.27 from 82 to 70 Total Balance:  100000.00
Thread[Thread-77,5,main]     297.15 from 77 to 44 Total Balance:  100000.00
```

可以看到程序创建了很多的线程。