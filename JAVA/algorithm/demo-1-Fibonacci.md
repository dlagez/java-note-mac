

这个数列从第3项开始，每一项都等于前两项之和。



解法：

```java
int fib(int n) {
    if (n == 2 || n == 1) 
        return 1;
    int prev = 1, curr = 1;
    for (int i = 3; i <= n; i++) {
        int sum = prev + curr;
        prev = curr;
        curr = sum;
    }
    return curr;
}
```

理解：

代码很简单，两个数相加，等于下一个数。

需要注意的是循环的时候是从3开始，因为第一第二个数已经给出来了。然后一步一步加下去即可。