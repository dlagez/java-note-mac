| scala                      | case                 | 状态 | 注释 | assign  |
| -------------------------- | -------------------- | ---- | ---- | ------- |
| RptFgProcess.scala         | rpt_fg_wrk_ord       |      |      | roc     |
|                            | rpt_po_es_union      |      |      | roc     |
|                            | rpt_fg_wrk_ord       |      |      | roc     |
|                            | rpt_po_es_union      |      |      | xiewang |
| RPT_hp_ia_ap_xact_t1.scala | RPT_hp_ia_ap_xact_t1 |      |      | xiewang |

问题记录：

64行：df是哪个表并不知道。**还没建表**

比如说我们几个表中取数据，a表和b表。所以我们写了一个select语句。

```SQL
select * 
from a left join b on a.id = b.id AND c.id = a.i
```

这个时候我能在on这个关键词后面使用c表的字段吗？比如 c.id = a.id。  

79行：df 不知道从哪冒出来的。**直接使用df1**

总结：

学习了vue 的使用及其生态系统router、store、vuex、axios

存储过程的开发。

准备深度学习相关知识的分享。主要是我们目前的方向，图像分类，目标检测，生成对抗网络。
