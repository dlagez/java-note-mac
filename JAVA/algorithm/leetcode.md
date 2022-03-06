#### [209. 长度最小的子数组](https://leetcode-cn.com/problems/minimum-size-subarray-sum/)

滑动窗口.

思路：首先创建一个窗口，这个窗口里面值的和在小于目标值时会一直扩大，end向后移动。在移动的过程中，如果数组的和大于目标值了，说明符合题意了。记录他的长度，然后将start后移，用来减小数组和和，直到这个数组的和小于目标值。然后继续右移end的值。

```java
package leetcode;

public class demo209 {
    public static void main(String[] args) {
        int s = 30;
        int[] nums = {1, 2, 3, 4, 5, 3, 4, 2, 6,4, 5, 6, 10, 6, 6, 9, 5, 6, 8, 0, 9};
        demo209 demo209 = new demo209();
        int i = demo209.minSubArrayLen(s, nums);
        System.out.println("最小子数组长度为： " + i);
    }
    public int minSubArrayLen(int s, int[] nums) {
        int n = nums.length;
        if (n == 0) return 0;
        int ans = Integer.MAX_VALUE;
        int start = 0, end = 0;
        int sum = 0;
        // 这里需要遍历整个数组
        while (end < n) {
            // 计算start到end之间的数值和
            sum += nums[end];
            // 当sum大于目标值s时，满足题目的需求，记录此时的数组长度。如果比已知最小的长度短，则更新ans的值。
            // 最后将左边的窗口右移。
            while (sum >= s) {
                ans = Math.min(ans, end - start + 1);
                // 如果sum大于s，这里会一直将左窗口右移，直到数组的和小于s。
                sum -= nums[start];
                start++;
            }
            end++;
        }
        // 这个ans初始化时就是Integer.MAX_VALUE，如果没有改变的话说明这个问题没有解
        return ans == Integer.MAX_VALUE ? 0 : ans;
    }
}

```



第一遍将`while(sum >= target)`写成`while(ans >= target)`导致错误。

第二遍循环需要找到的是数组和小于目标值的情况。ans表示最短数组的长度。

#### [217. 存在重复元素](https://leetcode-cn.com/problems/contains-duplicate/)

暴力解法。

```java
public class demo217 {
    public static void main(String[] args) {
        int[] nums = {1,2,3,1};
        demo217 demo217 = new demo217();
        boolean b = demo217.containsDuplicate(nums);
        System.out.println("是否包含重复： " + b);
    }

    public boolean containsDuplicate(int[] nums) {
        for (int i = 0; i < nums.length; i++) {
            for (int j = i + 1; j < nums.length; j++) {
                if (nums[i] == nums[j]) {
                    return true; 
                }
            }
        }
        return false;
    }
}
```



使用hash表，如果插入时返回false，说明hash表里面已经存在了这个值，说明这个数组里面有值时重复的。

```java
public boolean containsDuplicate2(int[] nums) {
        HashSet<Integer> set = new HashSet<>();
        for (int x : nums) {
            if (!set.add(x)) {
                return true;
            }
        }
        return false;
    }
```

return直接返回。会跳出两层循环



#### [76. 最小覆盖子串](https://leetcode-cn.com/problems/minimum-window-substring/)

要解决这个问题，首先得知道几个常用的数据结构，将hashmap转换成entryset，数据不会改变，就是改变输出形式。看下面的代码示例。

```java
// Java code to illustrate the entrySet() method
import java.util.*;

public class Hash_Map_Demo {
	public static void main(String[] args)
	{

		// Creating an empty HashMap
		HashMap<Integer, String> hash_map = new HashMap<Integer, String>();

		// Mapping string values to int keys
		hash_map.put(10, "Geeks");
		hash_map.put(15, "4");
		hash_map.put(20, "Geeks");
		hash_map.put(25, "Welcomes");
		hash_map.put(30, "You");

		// Displaying the HashMap
		System.out.println("Initial Mappings are: " + hash_map);

		// Using entrySet() to get the set view
		System.out.println("The set is: " + hash_map.entrySet());
	}
}

Initial Mappings are: {20=Geeks, 25=Welcomes, 10=Geeks, 30=You, 15=4}
The set is: [20=Geeks, 25=Welcomes, 10=Geeks, 30=You, 15=4]
```



使用滑动窗口的方法解决即可。

需要注意的问题是：当滑动窗口内的宽度小于目标字符串的长度时，需要将右指针移动到左指针字符串长度的位置。

```java
public class demo76 {
    public static void main(String[] args) {
        demo76 demo76 = new demo76();
        String s = demo76.minWindow("abaacbab", "abc");
        System.out.println("最小子字符串是：" + s);
    }

    Map<Character, Integer> ori = new HashMap<Character, Integer>();
    Map<Character, Integer> cnt = new HashMap<Character, Integer>();

    public String minWindow(String s, String t) {
        int tLen = t.length();
        // 将t中的全部字符解析出来放在ori中
        for (int i = 0; i < tLen; i++) {
            char c = t.charAt(i);
            ori.put(c, ori.getOrDefault(c, 0) + 1);
        }
        // 定义两个指针来控制滑动窗口的位置
        int l = 0, r = -1;
        // 这个len是用来记录最小子串长度的
        int len = Integer.MAX_VALUE, ansL = -1, ansR = -1;
        int sLen = s.length();
        while (r < sLen) {
            ++r;
            if (r < sLen && ori.containsKey(s.charAt(r))) {
                // 如果右指针这个字符属于目标字符，就把它放进cnt里面
                cnt.put(s.charAt(r), cnt.getOrDefault(s.charAt(r), 0) + 1);
            }
            // 将左指针右移，并将对应的元素移除
            while (check() && l <= r) {
                // 更新最小长度，并记录最小长度的子字符串
                if (r - l + 1 < len) {
                    len = r - l + 1;
                    ansL = l;
                    ansR = l + len;
                }
                // 如果左指针指向的这个字符串为目标字符串，则将它从cnt中移除。
                if (ori.containsKey(s.charAt(l))) {
                    cnt.put(s.charAt(l), cnt.getOrDefault(s.charAt(l), 0) - 1);
                }
                ++l;
            }
        }
        return ansL == -1 ? "" : s.substring(ansL, ansR);
    }
    // 检查滑动窗口包含的字符串是否包含所有的字符。
    public boolean check() {
        Iterator iter = ori.entrySet().iterator();
        while (iter.hasNext()) {
            Map.Entry entry = (Map.Entry) iter.next();
            Character key = (Character) entry.getKey();
            Integer val = (Integer) entry.getValue();
            if (cnt.getOrDefault(key, 0) < val) {
                return false;
            }
        }
        return true;
    }
}

```

复杂度分析：



时间复杂度：最坏的情况下左右指针对s的每个元素都遍历一遍，hash表对s中的每个元素各插入、删除一次。对t中的元素各插入一次。

每次检查是否可行会遍历整个t的hash表，hash表的大小与字符集大小有关，设字符集大小为C。这里的C表示t的长度。

O（C·｜s｜+｜t｜）

空间复杂度：这里只使用了两张hash表作为辅助空间，每张hash表最多不会存放超过字符集大小的键值对，空间复杂度为：O（C）。



#### [58. 最后一个单词的长度](https://leetcode-cn.com/problems/length-of-last-word/)

这里要注意一个事情：字符使用`' '`来表示。

```java
class Solution {
    public int lengthOfLastWord(String s) {
      	// 这里是length-1，不然会越界
        int index = s.length() - 1;
        while (s.charAt(index) == ' ') {
            index--;
        }
        int wordLength = 0;
      	// 若果对应 ‘a’的情况，必须加上 index >= 0 因为它的前面没有空格了。
        while (index >= 0 && s.charAt(index) != ' ') {
            wordLength++;
            index--;
        }
        return wordLength;
    }
}
```

