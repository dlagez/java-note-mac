题目链接：https://leetcode-cn.com/problems/koko-eating-bananas/

吃香蕉问题，猴哥吃香蕉的速度越快，消耗的时间越少，所以它是一个单调递减的函数。

在这个函数上找满足要求的最小时间。就可以用二分搜索来解决问题。

1.首先我们需要定一个函数，它可以求出吃的速度和吃完所需要时间的关系。

```java
// 吃完需要的时间， piles表示N堆香蕉，x表示吃香蕉的速度
    int f(int[] piles, int x) {
        int hours = 0;
        for (int i = 0; i < piles.length; i++) {
            // 吃其中一堆，如果还有剩余的，需要下一个小时再吃
            hours += piles[i] / x;
            // 这里加上吃剩下的
            if (piles[i] % x > 0) {
                hours++;
            }
        }
        return hours;
    }
```

2.根据这个函数我们可以使用二分搜索了。我们二分搜索的前提也是单调递增或者单调递减。

```java
    public int minEatingSpeed(int[] piles, int H) {
        int left = 1;
        int right = 1000000000 + 1;

        while (left < right) {
            int mid = left + (right - left) / 2;
            if (f(piles, mid) == H) {
                right = mid;
            } else if (f(piles, mid) < H) {
                right = mid;
            } else  if (f(piles, mid) > H) {
                left = mid + 1;
            }
        }
        return left;
    }
```

在普通的二分搜索中，mid可以直接算出来。这里只是定义了一个中间函数，mid的值需要使用这个函数计算出来。