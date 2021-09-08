### 冒泡排序：

- 比较相邻的元素，将较大的数放在后面。
- 一次循环就将最大的数放在了最后面。
- 每次循环需要比较的数都会减少一个（减少最后面最大的数）

```java
import java.util.*;
public class Solution {
    public int[] MySort (int[] arr) {
        // write code here
        if (arr.length == 0) return arr;
        for (int i = 0; i < arr.length -1; i++) {
            for (int j = 0; j < arr.length-1-i; j++) {
                if (arr[j+1] < arr[j]) {
                    int temp = arr[j];
                    arr[j] = arr[j+1];
                    arr[j+1] = temp;
                }
            }
        }
        return arr;
    }
}
```

复杂度：**T(n) = O(n2)**  由于循环了两次。复杂度为二次方

### 选择排序：

和冒泡排序差不多，将最小的数放在最前面。

```java
import java.util.*;
public class Solution {
    public int[] MySort (int[] arr) {
        // write code here
        if (arr.length == 0) return arr;
        for (int i = 0; i < arr.length; i++) {
            int minIndex = i;
            for (int j = i; j < arr.length; j++) {  
                if (arr[j] < arr[minIndex]) {  // 找出最小的数
                    minIndex = j; // 保存最小的数的索引
                }
            }
            if (minIndex != i) {
                int tmp = arr[i];
                arr[i] = arr[minIndex];
                arr[minIndex] = tmp;    
            }
        }
        return arr;
    }
}
```

### 快速排序：

```java
public class QuickSort implements IArraySort {

    @Override
    public int[] sort(int[] sourceArray) throws Exception {
        // 对 arr 进行拷贝，不改变参数内容
        int[] arr = Arrays.copyOf(sourceArray, sourceArray.length);

        return quickSort(arr, 0, arr.length - 1);
    }

    private int[] quickSort(int[] arr, int left, int right) {
        if (left < right) {
            int partitionIndex = partition(arr, left, right);
            quickSort(arr, left, partitionIndex - 1);
            quickSort(arr, partitionIndex + 1, right);
        }
        return arr;
    }

    private int partition(int[] arr, int left, int right) {
        // 设定基准值（pivot）
        int pivot = left;
        int index = pivot + 1;
      	// index始终指向第一个大于基准数的索引
        for (int i = index; i <= right; i++) {
            if (arr[i] < arr[pivot]) {
                swap(arr, i, index);
                index++;
            }
        }
      	// 将最后一个小于基准数的索引与基准数交换
        swap(arr, pivot, index - 1);
        return index - 1;
    }

    private void swap(int[] arr, int i, int j) {
        int temp = arr[i];
        arr[i] = arr[j];
        arr[j] = temp;
    }

}
```

口述快速排序的过程：

快速排序运用了分治的思想，拿到要进行排序的数组，我们首先确定一个基准数，一般我使用的是数组的第一个数，然后以这个数为基准，将比他小的数放在它的左边，比它大的数放在它的右边。然后以这个为基准，将数组分为左右两部份，循环使用这个方法对数组进行排序即可。

里面有一些小的细节，怎么移动数字。我使用了快慢指针，首先定义两个指针都指向基准数的后一个数字。使用for循环遍历这个数组。快指针每次循环值加一，当遇到数字比基准数字大的数，就将快指针和慢指针的值交换，慢指针值加一。这样就保证了快指针后面的数都比基准数字大，快慢指针之间的数字都比基准数字小。最后将基准数字，也就是数组的第一个数字与快指针索引减一的数字交换即可。
