| scala                      | case                 | 状态 | 注释 | assign  |
| -------------------------- | -------------------- | ---- | ---- | ------- |
| RptFgProcess.scala         | rpt_fg_wrk_ord       |      |      | roc     |
|                            | rpt_po_es_union      |      |      | roc     |
|                            | rpt_fg_wrk_ord       |      |      | roc     |
|                            | rpt_po_es_union      |      |      | xiewang |
| RPT_hp_ia_ap_xact_t1.scala | RPT_hp_ia_ap_xact_t1 |      |      | xiewang |

问题记录：

64行：df是哪个表并不知道。

65行：没有指明连接类型。

```
.join(fg_max_df, fg_max_df.col("wrk_ordid") === df.col("wrk_ord_id") && fg_max_df.col("cost_ctr") === df.col("cst_cntr_cd"))
```

79行：df 不知道从哪冒出来的。
