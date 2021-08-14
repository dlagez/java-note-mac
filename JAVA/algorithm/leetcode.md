题目：

https://leetcode-cn.com/problems/3sum/
解答：

https://leetcode-cn.com/problems/3sum/solution/hua-jie-suan-fa-15-san-shu-zhi-he-by-guanpengchn/

标签：数组遍历
首先对数组进行排序，排序后固定一个数 nums[i]nums[i]nums[i]，再使用左右指针指向 nums[i]nums[i]nums[i]后面的两端，数字分别为 nums[L]nums[L]nums[L] 和 nums[R]nums[R]nums[R]，计算三个数的和 sumsumsum 判断是否满足为 000，满足则添加进结果集
如果 nums[i]nums[i]nums[i]大于 000，则三数之和必然无法等于 000，结束循环
如果 nums[i]nums[i]nums[i] == nums[i−1]nums[i-1]nums[i−1]，则说明该数字重复，会导致结果重复，所以应该跳过
当 sumsumsum == 000 时，nums[L]nums[L]nums[L] == nums[L+1]nums[L+1]nums[L+1] 则会导致结果重复，应该跳过，L++L++L++
当 sumsumsum == 000 时，nums[R]nums[R]nums[R] == nums[R−1]nums[R-1]nums[R−1] 则会导致结果重复，应该跳过，R−−R--R−−
时间复杂度：O(n2)O(n^2)O(n2)，nnn 为数组长度

```java
class Solution {
    public static List<List<Integer>> threeSum(int[] nums) {
        List<List<Integer>> ans = new ArrayList();
        int len = nums.length;
        if(nums == null || len < 3) return ans;
        Arrays.sort(nums); // 排序
        for (int i = 0; i < len ; i++) {
            if(nums[i] > 0) break; // 如果当前数字大于0，则三数之和一定大于0，所以结束循环
            if(i > 0 && nums[i] == nums[i-1]) continue; // 去重
            int L = i+1;
            int R = len-1;
            while(L < R){
                int sum = nums[i] + nums[L] + nums[R];
                if(sum == 0){
                    ans.add(Arrays.asList(nums[i],nums[L],nums[R]));
                    while (L<R && nums[L] == nums[L+1]) L++; // 去重
                    while (L<R && nums[R] == nums[R-1]) R--; // 去重
                    L++;
                    R--;
                }
                else if (sum < 0) L++;
                else if (sum > 0) R--;
            }
        }        
        return ans;
    }
}

```



56.合并区间 https://leetcode-cn.com/problems/merge-intervals/

解答

```java
class Solution {
    public int[][] merge(int[][] intervals) {
        if (intervals.length == 0) {
            return new int[0][2];
        }
        Arrays.sort(intervals, new Comparator<int[]>() {
            public int compare(int[] interval1, int[] interval2) {
                return interval1[0] - interval2[0];
            }
        });
        List<int[]> merged = new ArrayList<int[]>();
        for (int i = 0; i < intervals.length; ++i) {
            int L = intervals[i][0], R = intervals[i][1];
            if (merged.size() == 0 || merged.get(merged.size() - 1)[1] < L) {
                merged.add(new int[]{L, R});
            } else {
                merged.get(merged.size() - 1)[1] = Math.max(merged.get(merged.size() - 1)[1], R);
            }
        }
        return merged.toArray(new int[merged.size()][]);
    }
}


作者：LeetCode-Solution
链接：https://leetcode-cn.com/problems/merge-intervals/solution/he-bing-qu-jian-by-leetcode-solution/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
```
