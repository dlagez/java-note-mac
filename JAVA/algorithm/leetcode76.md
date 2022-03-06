其他部分可以先不用看。看这部分的代码即可，它的两个while循环控制着左右指针的移动。

![image-20220222211932052](https://cdn.jsdelivr.net/gh/dlagez/img@master/image-20220222211932052.png)

外层的while控制右指针的移动。无论左指针怎么移动，右指针都会从左到右的遍历一遍。

如上图：右指针遍历到C之前，都不会进入到第二个while循环。因为不满足条件，在窗口中并没有包含所有的目标字符。直到遍历到C时（这时左指针还在开始的位置，就是A），窗口内包含了所有的目标字符（ABC），这时第二个循环的条件满足了，就进入到了第二个循环。

```
while (r < sLen) {
    ++r;
    if (r < sLen && ori.containsKey(s.charAt(r))) {
        // 如果右指针这个字符属于目标字符，就把它放进cnt里面
        cnt.put(s.charAt(r), cnt.getOrDefault(s.charAt(r), 0) + 1);
    }
    // 将左指针右移，并将对应的元素移除
    while (check() && l <= r) {
        // 更新最小长度
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
```

第一次进入第二个循环：

｜abaac｜bab，第一个竖线是左指针，第二个竖线是右指针。由于第二个循环控制左指针，左指针会一直向右移动（此时右指针不动），直到这里左指针走到这 ab｜aac｜bab，这个情况下滑动窗口已经不满足包含所有目标字符的条件了。就不会进入到第二个循环。此时记录最小子字符串的长度为4。

![Untitled](https://cdn.jsdelivr.net/gh/dlagez/img@master/Untitled.png)

第二次进入第二个循环：

第二次循环是这个样子的 ab | aacb | ab，此时是满足条件的，但是不会更新最小长度，因为他的长度还是4，上一次最小长度也是4。此时左指针向右移动aba ｜ acb | ab，此时长度为3，比4小，会更新最小长度

![Untitled 2](https://cdn.jsdelivr.net/gh/dlagez/img@master/Untitled%202.png)