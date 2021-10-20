由于有五天的数学建模。所以这里停了五天。

今天搞一下动态规划。

求斐波那契数列，很简单，一个递归方法即可。

```
int fib(int N) {
	if (N == 1 || N == 2) return 1;
	return fib(N-1) + fib(N - 2);
}
```

![图片](day4.assets/640)

这里有个问题，比如我们要求fib(20)，它等于fib(19)+fib(18)。所以他两需要被计算出来。而fib(19)=fib(18)+fib(17)。这里就有一个问题了。我们把fib(18)计算了两遍。并且求他需要的时间很长。

解决这个问题的方法是这样的。我们定义一个备忘录。如果要计算fib(18)的值。我们可以先在这个备忘录里面查询。如果已经计算了一遍我们就不需要再计算一遍了。只需要将备忘录里面的数据拿出来即可。

```java
int fib(int N) {
    if (N < 1) return 0;
    // 备忘录全初始化为 0
    vector<int> memo(N + 1, 0);
    // 初始化最简情况
    return helper(memo, N);
}

int helper(vector<int>& memo, int n) {
    // base case 
    if (n == 1 || n == 2) return 1;
    // 已经计算过，从备忘录查询值
    if (memo[n] != 0) return memo[n];
    memo[n] = helper(memo, n - 1) + 
                helper(memo, n - 2);
    return memo[n];
}
```

反向思维，我们反正是要将计算的结果存储在这张备忘录上。我们求大的数首先要求出小的数。不如直接从小到大的求更加的方便。

```java
int fib(int N) {
	vector<int> dp(N + 1, 0);
	dp[1] = d[2] = 1
	for (int i = 3; i <= N; i++) {
		dp[i] = dp[i-1] + dp[i-2];
	}
	return dp[N];
}
```

上面的代码还可以优化。将空间复杂度降为0；

不需要表。直接一层一层的往上求即可。



凑零钱问题：

