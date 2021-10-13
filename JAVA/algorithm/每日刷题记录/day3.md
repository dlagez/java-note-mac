今天准备把链表的主要操作刷完。包括：单链表的中点、判断链表是否包含环、两个链表是否相交。有时间再看看滑动窗口。

开干！

单链表的中点：没看答案，自己写一次通过。

思路：双指针快慢节点，快节点时慢节点速度的两倍，当快节点走到链表的终点的时候，慢节点刚刚好到中间。

注意⚠️：快节点要走两步。所以需要判断p2.next不为空，这样p2走两步不会报错。

```java
class Solution {
    public ListNode middleNode(ListNode head) {
        ListNode p1 = head, p2 = head;
        while (p2 != null && p2.next != null) {
            p1 = p1.next;
            p2 = p2.next.next;
        }
        return p1;
    }
}
```

单链表判断环：看了下答案，中间卡顿了。

思路：如果有环，快指针肯定一追上满指针，他两相遇。判断他两箱等的时候返回true即可。

```java
public class Solution {
    public boolean hasCycle(ListNode head) {
        ListNode slow = head, fast = head;
        while (fast != null && fast.next != null) {
            slow = slow.next;
            fast = fast.next.next;
            // 如果有环，他两肯定会相遇
            if (slow == fast) {
                return true;
            }
        }
    // 如果fast指针遇到终点了说明没有环
    return true;
    }
}
```

进阶版，找到环的起点。

```java
public class Solution {
    public ListNode detectCycle(ListNode head) {
        ListNode slow = head, fast = head;
        while(fast != null && fast.next != null) {
            slow = slow.next;
            fast = fast.next;
            // 他两相遇就停下来
            if (slow == fast) break;
        }

        // 如果fast遇到空值，说明没有环
        while (fast == null && fast.next == null) {
            return false;
        }

        // 将其中一个设置到起点，他两相遇的地方就是环的起点。
        slow = head;
        while (slow != fast) {
            slow = slow.next;
            fast = fast.next;
        }
        return slow;
    }
}
```

找到两条链表的交点：

注意⚠️：这里的p1走完了之后，将节点指向AB原节点时，不能带p1.next = headB，会报空指针异常。因为p1此时已经是空指针了，直接将指针指向headB即可。

```java
public class Solution {
    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        ListNode p1 = headA, p2 = headB;
        while (p1 != p2) {
            if (p1 == null) p1 = headB;
            else p1 = p1.next;
            if (p2 == null) p2 = headA;
            else p2 = p2.next; 
        }
        return p1;
    }
}
```

