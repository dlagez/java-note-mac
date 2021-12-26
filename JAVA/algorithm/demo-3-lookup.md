## 暴力搜索

```java
public static int search(int[] a, int key) {
    for (int i = 0, length = a.length; i < length; i++) {
        if (a[i] == key)
            return i;
    }
    return -1;
}
```



## 基本的二分查找

```java
int binarySearch(int[] nums, int target) {
	int left = 0, right = nums.length - 1;
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if(nums[mid] == target) {
            return mid;
        } else if (nums[mid] < target) {
            left = mid + 1;
        } else if (nums[mid] > target) {
            right = mid - 1;
        }
        return -1;
    }
}
```

这里有几个问题要高清楚：

### 问题一：

1、**为什么 while 循环的条件中是 <=，而不是 <**？

答：因为初始化`right`的赋值是`nums.length - 1`，即最后一个元素的索引，而不是`nums.length`。

`while(left <= right)`的终止条件是`left == right + 1`，写成区间的形式就是`[right + 1, right]`，或者带个具体的数字进去`[3, 2]`，可见**这时候区间为空**，因为没有数字既大于等于 3 又小于等于 2 的吧。所以这时候 while 循环终止是正确的，直接返回 -1 即可。

`while(left < right)`的终止条件是`left == right`，写成区间的形式就是`[left, right]`，或者带个具体的数字进去`[2, 2]`，**这时候区间非空**，还有一个数 2，但此时 while 循环终止了。也就是说这区间`[2, 2]`被漏掉了，索引 2 没有被搜索，如果这时候直接返回 -1 就是错误的。



当然，如果你非要用`while(left < right)`也可以，我们已经知道了出错的原因，就打个补丁好了：

```java
    //...
    while(left < right) {
        // ...
    }
    return nums[left] == target ? left : -1;
```

### 问题二：

**2、为什么****`left = mid + 1`，`right = mid - 1`？我看有的代码是`right = mid`或者`left = mid`，没有这些加加减减，到底怎么回事，怎么判断**？

答：这也是二分查找的一个难点，不过只要你能理解前面的内容，就能够很容易判断。

刚才明确了「搜索区间」这个概念，而且本算法的搜索区间是两端都闭的，即`[left, right]`。那么当我们发现索引`mid`不是要找的`target`时，下一步应该去搜索哪里呢？

当然是去搜索`[left, mid-1]`或者`[mid+1, right]`对不对？**因为`mid`已经搜索过，应该从搜索区间中去除**。

### 问题三：

**3、此算法有什么缺陷**？

比如说给你有序数组`nums = [1,2,2,2,3]`，`target`为 2，此算法返回的索引是 2，没错。但是如果我想得到`target`的左侧边界，即索引 1，或者我想得到`target`的右侧边界，即索引 3，这样的话此算法是无法处理的。



## 二分查找寻找左侧边界

```java
int left_bound(int[] nums, int target) {
	int left = 0, right = nums.length - 1;
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (nums[mid] < target) {
            left = mid + 1;
        } else if (nums[mid] > target) {
            right = mid - 1;
        } else if (nums[mid] == target) {
            // 搜索区间变成[left, mid - 1] 收缩搜索区间
            right = mid - 1;
        }
    }
    // 检查出界的情况，由于需要返回left，所以需要检查left是否越界。
    if (left > nums.length || nums[left] != target) {
        return -1;
    }
    return left;
}
```

**为什么该算法能够搜索左侧边界**？

答：关键在于对于`nums[mid] == target`这种情况的处理：

```
    if (nums[mid] == target)
        right = mid;
```

可见，找到 target 时不要立即返回，而是缩小「搜索区间」的上界`right`，在区间`[left, mid)`中继续搜索，即不断向左收缩，达到锁定左侧边界的目的。



**检查出界的情况为什么只检查left**

由于需要返回left，所以需要检查left是否越界。right也会越界，但是无所谓了，right越界了也不会引发问题。





