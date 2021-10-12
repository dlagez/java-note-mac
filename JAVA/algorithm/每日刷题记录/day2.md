BFS

**问题的本质就是让你在一幅「图」中找到从起点`start`到终点`target`的最近距离**

**BFS 找到的路径一定是最短的，但代价就是空间复杂度比 DFS 大很多**

leetcode:[二叉树的最小深度](https://leetcode-cn.com/problems/minimum-depth-of-binary-tree/)

框架

```java
// 计算从起点 start 到终点 target 的最近距离
int BFS(Node start, Node target) {
    Queue<Node> q; // 核心数据结构
    Set<Node> visited; // 避免走回头路

    q.offer(start); // 将起点加入队列
    visited.add(start);
    int step = 0; // 记录扩散的步数

    while (q not empty) {
        int sz = q.size();
        /* 将当前队列中的所有节点向四周扩散 */
        for (int i = 0; i < sz; i++) {
            Node cur = q.poll();
            /* 划重点：这里判断是否到达终点 */
            if (cur is target)
                return step;
            /* 将 cur 的相邻节点加入队列 */
            for (Node x : cur.adj())
                if (x not in visited) {
                    q.offer(x);
                    visited.add(x);
                }
        }
        /* 划重点：更新步数在这里 */
        step++;
    }
}
```

题解：

// 这里的方法是这样的。在for循环里面，会遍历这一层的节点，将这一层的节点全部添加到q里面。

只要碰到了最小深度，左右节点都是空的。就直接返回即可。

```java
class Solution {
    public int minDepth(TreeNode root) {
        if (root == null) return 0;
        Queue<TreeNode> q = new LinkedList<>();
        q.offer(root);
        int depth = 1;

        while (!q.isEmpty()) {
            int sz = q.size();
            for (int i = 0; i < sz; i++) {
                TreeNode cur = q.poll();
                // 判断是否到达终点
                if (cur.left == null && cur.right == null) {
                    return depth;
                }
                if (cur.left != null) {
                    q.offer(cur.left);
                }
                if (cur.right != null) {
                    q.offer(cur.right);
                }
            }
            depth++;
        }
        return depth;
    }
}
```

再复习下链表的操作。

合并两个有序的链表:

```java
class Solution {
    public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
      // 这里定义dummy是为了返回方便，因为p操作新增节点，p一直在next往后走，直接返回p会丢失信息  
      ListNode dummy = new ListNode(-1), p = dummy;
      // 这里是为了不破坏原节点  
      ListNode p1 = l1, p2 = l2;

        while(p1 != null && p2 != null) {
            if (p1.val < p2.val) {
                p.next = p1;
                p1 = p1.next;
            } else {
                p.next = p2;
                p2 = p2.next;
            }
            p = p.next;
        }

        if (p1 != null) {
            p.next = p1;
        }
        if (p2 != null) {
            p.next = p2;
        }
      // 返回时很方便，dummy记录p所有的节点。因为他和p共同指向一个对象。
        return dummy.next;
     }
}
```

合并n和有序列表，里面的方法是这样的。我们合并两个列表时只需要必将两个数的大小就行。这里有n个列表，我们可以使用最小堆来简化操作，比较大小的任务交给最小堆来完成，我们只需要每次从最小堆里面取出最小的元素即可。

```java
class Solution {
    public ListNode mergeKLists(ListNode[] lists) {
        if (lists.length == 0) return null;
        // 设置一个虚拟节点，用来记录节点加入信息
        ListNode dummy = new ListNode(-1);
        ListNode p = dummy;

        PriorityQueue<ListNode> pq = new PriorityQueue<>(
            lists.length, (a, b) -> (a.val - b.val)
        );
        // 这里pq.add的元素已经由小到大排序好了
        for (ListNode head : lists) {
            if (head != null) {
                pq.add(head);
            }
        }

        while(!pq.isEmpty()) {
            // 取出最小的元素，并从最小堆中删除
            ListNode node = pq.poll();
            p.next = node;
            // 取出节点时再将后面的节点放入到最小堆
            if (node.next != null) {
                pq.add(node.next);
            }
            p = p.next;
        }

        return dummy.next;
    }
}
```

找倒数第n个节点

```java
ListNode findFromEnd(ListNode head, int k) {
  ListNode p1 = head;
  for (int i = 0; i < k; i++) {
    p1 = p1.next;
  }
  ListNode p2 = head;
  while (p1 != null) {
    p2 = p2.next;
    p1 = p1.next;
  }
  return p2;
}
```

删除倒数第n个节点就好办了

```java
    public ListNode removeNthFromEnd(ListNode head, int n) {
        ListNode dummy = new ListNode(-1);
        dummy.next = head;
        // 这里是倒着找的，找到n的前一个，即可删除第n个数。
        ListNode x = findFromEnd(dummy, n + 1);
        x.next = x.next.next;
        return dummy.next;
    }
```

