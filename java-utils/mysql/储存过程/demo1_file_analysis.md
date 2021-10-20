#### sql

首先这个sql文件里面的代码是一个存储过程。这个存储过程的作用是将联合和转换的数据插入目标表。

首先使用了关键字 with as ：定义了一个SQL片段，该SQL片段会被整个SQL语句所用到，文件中sql片段里面包含了**数据的选择、转换和连接**，也就是包含三个select语句的连接。

- 首先是将数据进行转换和连接。转换函数使用的是convert函数。这个函数的作用是在不同字符集之间转换数据，根据存储过程中的convert函数的语法，CONVERT(expr, type)语法包含一个表达式和一个类型值（用来指定结果的类型），生成指定类型的结果值。
- 这里使用了UNION将多个SELECT语句的结果合并到一个结果集中。使用UNION关键字需要注意UNION 内部的 SELECT 语句必须拥有相同数量的列。
- 用到了case when表达式。

sql文件的最后将插入到ING表中的数据查询了出来。应该用来查看插入到目标表中的数据，这个语句应该和scala代码没有关系。

#### get_pklist_and_col_mapping

首先这个函数的作用应该是定义三个列表，这三个列表里面的数据分别表示：第一个列表应该是选择出提供数据的表的名字（从这个表里面选择数据并转换），第二个列表里面的数据应该是选择列的表达式（定义选择出来的数据列名，并进行数据的转换和过滤），第三个列表应该是包含目标表的名字（将数据插入到这个表里面），它是通过match方法确定的。

get_pklist_and_col_mapping函数里面使用match方法来匹配case。当entity的target_table属性值为对应case就确定了需要选择的列。

每个case里面有三个list列表。

- pkList列表应该是对应三个select目标表。也就是从这三个表格里面选择数据。
- colExpr：定义每一列的表达式，对应sql语句的选择列的语句。
- trgPkList：应该是目标表吧，把数据插入到这些表里面。对应sql语句里面的 INSERT INTO 

#### ing_compass_union_all

这个方法应该是将三个select语句选择出来的数据进行连接。

这个方法传入四个参数，前两个spack，entity应该是操作sql语句的工具，list是表，load_jb_nr是控制条件，它的值为0和其他值的时候执行的语句不同。

这个方法里面定义了四个变量：

- 前三个变量是与表相关的变量，应该就是表的数据。
- 这个主要说一下get_rgn变量， 这个变量表达式应该是对应select语句里面的case when语句，存储程序的CASE语句实现了一个复杂的条件构造。这个get_rgn变量是匹配语句，它与withColumn这个方法联合使用，withColumn方法的作用是通过添加列或替换具有相同名称的现有列来返回新数据集。
- 将三个数据处理完之后，他们选择出来的数据就确定了，然后使用 df1.union(df2).union(df3) 表达式将他们的结果进行连接，对应sql语句的union关键字，将三个select语句选择的数据进行连接。

当load_jb_nr的值不为0的时候，会有一个判断，edl_crt_ts的值要大于等于get_compass_max_edl_ctr_ts函数的输出。

#### get_compass_max_edl_ctr_ts

这个方法应该是选择edl_crt_ts最大值。

#### get_compass_max_load_jb_nr

选择edl_job_sqn_nr的最大值



## 语句分析任务：

### expr [链接](https://sparkbyexamples.com/pyspark/pyspark-sql-expr-expression-function/)

是一个SQL函数，用于执行类似SQL的表达式

### withColumn函数：[链接](https://spark.apache.org/docs/3.2.0/api/scala/org/apache/spark/sql/Dataset.html#withColumn(colName:String,col:org.apache.spark.sql.Column):org.apache.spark.sql.DataFrame)

添加列，或者替换同名的列。

### 语句拆分分析：

提取出其中的sql表达式，就是三个if判断，if条件为true（选定的字段都为空），则使用指定的字段比如'BAUnknown'。

如果非空判断都为false的话。下面的语句会输出类似（'BAUnknown', 'M.BUSINESS_AREA', 'cc.cc_BUSINESS_AREA'）

```sql
if((cc.cc_BUSINESS_AREA is null and io.io_BUSINESS_AREA is null and M.BUSINESS_AREA is null),'BAUnknown',
  		if((cc.cc_BUSINESS_AREA is null and io.io_BUSINESS_AREA is null) ,M.BUSINESS_AREA,
    			if(cc.cc_BUSINESS_AREA is null,io.io_BUSINESS_AREA,cc.cc_BUSINESS_AREA)
  	)
)
```

使用withColumn将"bsn_ar_reorg_cd"和表达式得出的值合并，就是最终的结果。这里可能有个问题，如果表达式的值和"bsn_ar_reorg_cd"同名的话，新加的列会替换同名的列。

### 整个语句分析：

如果整个语句用于选择的，if判断都为flase的话，会将（'bsn_ar_reorg_cd'，'BAUnknown', 'M.BUSINESS_AREA', 'cc.cc_BUSINESS_AREA'）这四个列选择出来。

```scala
.withColumn("bsn_ar_reorg_cd", expr("if((cc.cc_BUSINESS_AREA is null and io.io_BUSINESS_AREA is null and M.BUSINESS_AREA is null),'BAUnknown',if((cc.cc_BUSINESS_AREA is null and io.io_BUSINESS_AREA is null) ,M.BUSINESS_AREA,if(cc.cc_BUSINESS_AREA is null,io.io_BUSINESS_AREA,cc.cc_BUSINESS_AREA)))"))
```

### 



```scala
package com.dxc.edl.ing.compass

import com.dxc.edl.util.BaseFunctions._
import org.apache.log4j.Logger
import org.apache.spark.sql.functions.{expr, lit, upper, when}
import org.apache.spark.sql.{DataFrame, SparkSession}

object IngCompassProcess extends Serializable {
  val logger: Logger = Logger.getLogger(getClass.getName)

  def get_pklist_and_col_mapping(entity: SourceEntityMappings): (List[String], List[String], List[String]) = {

    // 通过case表达式确定ret_val的值
    val ret_val: (List[String], List[String], List[String]) = entity.target_table.toLowerCase match {

      // Compass pklist and column mapping
      case "ing_cmpss_po" =>
        val pkList = List("purchase_order_number", "po_line_number", "rgn_typ")
        val colExpr = List("concat_ws('~',purchase_order_number,po_line_number,'COMPASS') as ky_id", "'COMPASS' as src_sys_nm", "purchase_order_number as po_nr", "purchase_order_type as po_typ_cd", "porg as porg_cd", "pgroup as prchg_grp_cd", "nvl(to_date(po_document_date, 'yyyyMMdd'),po_document_date) as po_dcmt_dt", "company_code as co_cd", "vendor_num as vndr_nr", "vendor_name as vndr_nm", "po_line_number as po_ln_itm_nr", "deletion_flag as dltn_flg", "material_group as mtrl_grp_cd", "material_group_description as mtrl_grp_dn", "nvl(to_date(po_last_modified_date, 'yyyyMMdd'),po_last_modified_date) as po_last_mfyd_dt", "material_number as mtrl_nr", "po_line_item_description as itm_dn", "po_plant as po_plnt_cd", "purchase_requisition_number as rqstn_nr", "pr_line_item_number as ln_itm_on_req_nr", "cost_object as fncl_mppg_cd", "profit_center as pft_cntr_cd", "business_area_fact as bsn_ar_fct_cd", "substr(gl_account, 1, 4) as gl_acct_nr", "project_id as proj_id", "wbs_id as wbs_cd", "po_requestor as rqstr_login_id", "nvl(po_item_local_amount, 0) as ln_itm_totl_lcl_curr_amt", "nvl(po_item_usd_amount, 0) as ln_itm_totl_usd_amt", "nvl(local_currency, 0) as lcl_curr_cd", "nvl(purchase_order_qty,0) as qty", "hours as hrs_qty", "unit_of_measure as of_unt_msr_cd", "payment_terms as pmnt_trm_cd", "requirement_tracking_number as rqmt_trkg_id", "contract_number as cctr_id", "gl_account as cctr_nm", "vendor_material_number as mfrr_prt_nr", "delivery_country as ctry_cd", "ccow_fte as ccow_fte_nr", "nvl(to_date(po_delivery_date, 'yyyyMMdd'),po_delivery_date) as po_dlvry_dt", "sl_order_code as sl_ord_cd", "sl_labor_key as lr_ky", "'Non core skill (commodity skill,100% outsourced)' as rtnl_for_otsrc_lr_txt", "po_version as po_vrsn_nr", "po_version_completion_flag as po_vrsn_cmpltn_flg", "ariba_id as ariba_id", "if(split(ariba_id, '-')[1] is null, ariba_id,split(ariba_id, '-')[0]) as pr_nr", "if(split('ariba_id', '-')[1] is null, 0 ,split('ariba_id', '-')[1]) as pr_ln", "rgn_typ as rgn_typ", "edl_crt_ts as edl_crt_ts", "edl_upd_ts as edl_upd_ts", "edl_src_sys_ky as edl_src_sys_ky", "edl_upd_tm_ky as edl_upd_tm_ky", "fl_nm as fl_nm", "edl_chcksm_tx as edl_chcksm_tx", "edl_job_sqn_nr as edl_job_sqn_nr")
        val trgPkList = List("ky_id", "rgn_typ")
        (pkList, colExpr, trgPkList)

      case "ing_cmpss_pr" =>
        val pkList = List("region", "requisition_number", "requisition_line_number", "rgn_typ")
        val colExpr = List("concat_ws('~',region ,requisition_number ,requisition_line_number) as ky_id", "acct_assignment_category as acct_asngmt_cgy_cd", "compass_business_area as cmpss_bsn_ar_nm", "currency as curr_cd", "country as ctry_cd", "nvl(to_date(delivery_date, 'yyyyMMdd'), to_date('2005-03-24')) as dlvry_dt", "nvl(to_date(create_date, 'yyyyMMdd'), to_date('2005-03-24')) as crc_dt", "fixed_vendor as fx_vndr_cd", "gl_account as gl_acct_nr", "item_category as itm_cgy_cd", "material_group as mtrl_grp_cd", "net_value_local as nt_vl_lcl_amt", "net_value_usd as usd_nt_vl_amt", "purchasing_group as prchg_grp_cd", "purchase_order_number as po_nr", "purchase_line_item as prch_ln_itm_cd", "requisition_number as rqstn_nr", "requisition_line_number as rqstn_ln_nr", "plant as plnt_cd", "purchasing_org as prchg_org_cd", "quantity as pr_qty", "requirements_tracking_number as rqrmnts_trkg_nr", "requisitioner as rqstnr_id", "hiring_manger as hreg_mgr_eml_addr_txt", "short_text as shrt_txt", "unit_of_measure as uom_cd", "wbs_id as wbs_id", "project_id as proj_id", "cost_object_id as cst_obj_id", "region as pr_atr_1_txt", "0 as pr_atr_2_txt", "if(instr(ariba_id, '-')=0,ariba_id,substr(ariba_id, 1, instr(ariba_id, '-')-1)) as pr_atr_3_txt", "if(instr(ariba_id, '-')=0, '0' , substr(ariba_id,1,instr(ariba_id, '-')+1)) as pr_atr_4_txt", "'0' as pr_atr_5_txt", "'0' as pr_atr_6_txt", "'0' as pr_atr_7_txt", "'0' as pr_atr_8_txt", "'0' as pr_atr_9_txt", "'ES' as co_id", "edl_crt_ts as edl_crt_ts", "edl_upd_ts as edl_upd_ts", "edl_src_sys_ky as edl_src_sys_ky", "edl_upd_tm_ky as edl_upd_tm_ky", "fl_nm as fl_nm", "edl_chcksm_tx as edl_chcksm_tx", "edl_job_sqn_nr as edl_job_sqn_nr")
        val trgPkList = List("ky_id")
        (pkList, colExpr, trgPkList)

      case "ing_cmpss_proj_dim" =>
        val pkList = List("project_id", "rgn_typ")
        val colExpr = List("project_id as ky_id", "'ES' as co_id", "project_id as proj_id", "project_description as proj_dn", "project_responsible_person_id as proj_rspnsbl_pers_id", "project_admin_resource_id as proj_admn_rsrc_id", "project_admin_name as proj_admn_nm", "project_financial_analyst_id as proj_fncl_anlst_id", "project_financial_analyst_name as proj_fncl_anlst_nm", "project_customer_id as proj_cust_id", "project_customer_name as proj_cust_nm", "deletion_flag as dltn_flg", "'COMPASS' as src_id", "contract_id as cntrct_id", "contract_name as cntrct_nm", "'' as cntrct_nr", "'' as proj_atr_1_cd", "'' as proj_atr_1_tx", "'' as proj_atr_2_cd", "'' as proj_atr_2_tx", "'' as proj_atr_3_cd", "'' as proj_atr_3_tx", "'' as proj_atr_4_cd", "'' as proj_atr_4_tx", "'' as proj_atr_5_cd", "'' as proj_atr_5_tx", "rgn_typ as rgn_typ", "edl_crt_ts as edl_crt_ts", "edl_upd_ts as edl_upd_ts", "edl_src_sys_ky as edl_src_sys_ky", "edl_upd_tm_ky as edl_upd_tm_ky", "fl_nm as fl_nm", "edl_chcksm_tx as edl_chcksm_tx", "edl_job_sqn_nr as edl_job_sqn_nr")
        val trgPkList = List("ky_id")
        (pkList, colExpr, trgPkList)

      case "ing_cmpss_wbs_dim" =>
        val pkList = List("wbs_id", "rgn_typ")
        val colExpr = List("wbs_id as ky_id", "'ES' as co_id", "wbs_id as wbs_cd", " wbs_description as wbs_dn", "wbs_business_group as  wbs_bsn_grp_cd", "deletion_flag as dltn_flg", "'COMPASS' as src_id", "contract_number as cntrct_nr", "wbs_id_without_msk as  wbs_atr_1_cd", "project_type_key as wbs_atr_1_tx", "project_type_text as wbs_atr_2_cd", "bpa as  wbs_atr_2_tx", "wbs_cost_center_post as wbs_atr_3_cd", "wbs_statistical_flag as  wbs_atr_3_tx", "wbs_task_order as wbs_atr_4_cd", "wbs_resp_cost_center as  wbs_atr_4_tx", "controlling_area as wbs_atr_5_cd", "'' as  wbs_atr_5_tx", "rgn_typ as rgn_typ", "edl_crt_ts as edl_crt_ts", "edl_upd_ts as edl_upd_ts", "edl_src_sys_ky as edl_src_sys_ky", "edl_upd_tm_ky as edl_upd_tm_ky", "fl_nm as fl_nm", "edl_chcksm_tx as edl_chcksm_tx", "edl_job_sqn_nr as edl_job_sqn_nr")
        val trgPkList = List("ky_id")
        (pkList, colExpr, trgPkList)

      // unmatched table
      case x =>
        logger.info(s"Inavlid Target Table : $x ")
        (List(), List(), List())

    }

    ret_val
  }

  def ing_compass_union_all(spark: SparkSession, entity: SourceEntityMappings, list:List[String], load_jb_nr:Int): DataFrame = {

    import spark.implicits._
    val imt_tab1 = SCM_IMT_TABLE(list.head)
    val imt_tab2 = SCM_IMT_TABLE(list(1))
    val imt_tab3 = SCM_IMT_TABLE(list(2))
    // 定义get_rgn 这个表达式应该是对应select语句里面的case when语句
    val get_rgn = when(upper($"fl_nm").contains("AMERICAS") || upper($"fl_nm").contains("AMS"), lit("AMS"))
      .when(upper($"fl_nm").contains("APJ"), lit("APJ")).when(upper($"fl_nm").contains("EMEA"), lit("EMEA"))
    if (load_jb_nr == 0) {
      // withColumn 通过添加列或替换具有相同名称的现有列来返回新数据集。
      val df1 = spark.sql(s"select * from $imt_tab1").withColumn("rgn_typ", get_rgn)
      val df2 = spark.sql(s"select * from $imt_tab2").withColumn("rgn_typ", get_rgn)
      val df3 = spark.sql(s"select * from $imt_tab3").withColumn("rgn_typ", get_rgn)
      df1.union(df2).union(df3)
    } else {

      val df1 = spark.sql(s"select * from $imt_tab1  where edl_crt_ts >= '${get_compass_max_edl_ctr_ts(spark, imt_tab1)}'").withColumn("rgn_typ", get_rgn)
      val df2 = spark.sql(s"select * from $imt_tab2  where edl_crt_ts >= '${get_compass_max_edl_ctr_ts(spark, imt_tab2)}'").withColumn("rgn_typ", get_rgn)
      val df3 = spark.sql(s"select * from $imt_tab3  where edl_crt_ts >= '${get_compass_max_edl_ctr_ts(spark, imt_tab3)}'").withColumn("rgn_typ", get_rgn)

      if(entity.target_table == "ing_cmpss_po") {
        df1.union(df2).union(df3).filter(expr("purchase_order_number not like 'X%' "))
      } else {
        df1.union(df2).union(df3)
      }
    }

  }

  def get_compass_max_edl_ctr_ts(spark: SparkSession, tbl_nm: String): String = {

    val num = spark.sql(s"select nvl(max(edl_crt_ts), '2018-01-01 00:00:00') mx_edl_crt_ts from $tbl_nm").collect().map(x => x(0).toString).head
    logger.info(s"$logString: Max of edl_crt_ts is $num")
    num
  }

  // 下面两个函数相当于重载函数
  def get_compass_max_load_jb_nr(spark: SparkSession, tbl_nm: String): Int = {
    // edl_job_sqn_nr，在case里面
    val num = spark.sql(s"select nvl(max(edl_job_sqn_nr), 0) ld_jb_nr from $tbl_nm").collect().map(x => x(0).toString).head.toInt
    logger.info(s"$logString: Max of edl_job_sqn_nr is $num")
    num
  }

  def get_compass_max_load_jb_nr(spark: SparkSession, list: List[String]): Int = {

    val imt_tab1 = SCM_IMT_TABLE(list.head)
    val imt_tab2 = SCM_IMT_TABLE(list(1))
    val imt_tab3 = SCM_IMT_TABLE(list(2))

    val ljn1 = spark.sql(s"select max(edl_job_sqn_nr) ld_jb_nr from $imt_tab1").collect().map(x => x(0).toString.toInt).head
    val ljn2 = spark.sql(s"select max(edl_job_sqn_nr) ld_jb_nr from $imt_tab2").collect().map(x => x(0).toString.toInt).head
    val ljn3 = spark.sql(s"select max(edl_job_sqn_nr) ld_jb_nr from $imt_tab3").collect().map(x => x(0).toString.toInt).head

    if (ljn1 < ljn2 && ljn1 < ljn3) ljn1 else if (ljn2 < ljn3) ljn2 else ljn3
  }


}
```



```sql
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
/*
* AUTHOR : srinivas.peddi@dxc.com
* Create date: 13-10-2021
* DESCRIPTION : This Stored Proc is for IngCompassProcess class. It inserts the unioned and transformed data into the ING target table
*/
CREATE PROC [scm_ing].[sp_ing_cmpss_po] AS
BEGIN  
SET NOCOUNT ON;  # 使返回的结果中不包含有关受 Transact-SQL 语句影响的行数的信息。
WITH CTE_UNION AS
(
SELECT 
concat_ws('~',purchase_order_number,po_line_number,'COMPASS') as [ky_id],
'COMPASS' as [src_sys_nm],
[purchase_order_number] as [po_nr],
[purchase_order_type] as [po_typ_cd],
[porg] as [porg_cd],
[pgroup] as [prchg_grp_cd],
convert(datetime2(0), po_document_date) as [po_dcmt_dt],
[company_code] as [co_cd],
[vendor_num] as [vndr_nr],
[vendor_name] as [vndr_nm],
[po_line_number] as [po_ln_itm_nr],
[deletion_flag] as [dltn_flg],
[material_group] as [mtrl_grp_cd],
[material_group_description] as [mtrl_grp_dn],
convert(datetime2(0), po_last_modified_date) as [po_last_mfyd_dt],
[material_number] as [mtrl_nr],
[po_line_item_description] as [itm_dn],
[po_plant] as [po_plnt_cd],
[purchase_requisition_number] as [rqstn_nr],
[pr_line_item_number] as [ln_itm_on_req_nr],
[cost_object] as [fncl_mppg_cd],
[profit_center] as [pft_cntr_cd],
[business_area_fact] as [bsn_ar_fct_cd],
substring(gl_account, 1, 4) as [gl_acct_nr],
[project_id] as [proj_id],
[wbs_id] as [wbs_cd],
[po_requestor] as [rqstr_login_id],
coalesce(po_item_local_amount, 0) as [ln_itm_totl_lcl_curr_amt],
coalesce(po_item_usd_amount, 0) as [ln_itm_totl_usd_amt],
coalesce(local_currency, 0) as [lcl_curr_cd],
coalesce(purchase_order_qty, 0) as [qty],
[hours] as [hrs_qty],
[unit_of_measure] as [of_unt_msr_cd],
[payment_terms] as [pmnt_trm_cd],
[requirement_tracking_number] as [rqmt_trkg_id],
[contract_number] as [cctr_id],
[gl_account] as [cctr_nm],
[vendor_material_number] as [mfrr_prt_nr],
[delivery_country] as [ctry_cd],
[ccow_fte] as [ccow_fte_nr],
convert(datetime2(0), po_delivery_date) as [po_dlvry_dt],
[sl_order_code] as [sl_ord_cd],
[sl_labor_key] as [lr_ky],
'Non core skill (commodity skill,100% outsourced)' as [rtnl_for_otsrc_lr_txt],
[po_version] as [po_vrsn_nr],
[po_version_completion_flag] as [po_vrsn_cmpltn_flg],
[ariba_id] as [ariba_id],
case when ariba_id like '%-%' then SUBSTRING(ariba_id, 0, charindex('-',ariba_id))
	 else ariba_id
end as [pr_nr],
case when ariba_id like '%-%' then SUBSTRING(ariba_id, charindex('-', ariba_id)+1, len(ariba_id))
	 else '0'
end as [pr_ln],
'AMS' as [rgn_typ], 
[edl_crt_ts] as [edl_crt_ts],
[edl_upd_ts] as [edl_upd_ts],
[edl_src_sys_ky] as [edl_src_sys_ky],
[edl_upd_tm_ky] as [edl_upd_tm_ky],
[fl_nm] as [fl_nm],
[edl_chcksm_tx] as [edl_chcksm_tx],
[edl_job_sqn_nr] as [edl_job_sqn_nr]
 FROM imt_compass_po_ams 
UNION
SELECT
concat_ws('~',purchase_order_number,po_line_number,'COMPASS') as [ky_id],
'COMPASS' as [src_sys_nm],
[purchase_order_number] as [po_nr],
[purchase_order_type] as [po_typ_cd],
[porg] as [porg_cd],
[pgroup] as [prchg_grp_cd],
convert(datetime2(0), po_document_date) as [po_dcmt_dt],
[company_code] as [co_cd],
[vendor_num] as [vndr_nr],
[vendor_name] as [vndr_nm],
[po_line_number] as [po_ln_itm_nr],
[deletion_flag] as [dltn_flg],
[material_group] as [mtrl_grp_cd],
[material_group_description] as [mtrl_grp_dn],
convert(datetime2(0), po_last_modified_date) as [po_last_mfyd_dt],
[material_number] as [mtrl_nr],
[po_line_item_description] as [itm_dn],
[po_plant] as [po_plnt_cd],
[purchase_requisition_number] as [rqstn_nr],
[pr_line_item_number] as [ln_itm_on_req_nr],
[cost_object] as [fncl_mppg_cd],
[profit_center] as [pft_cntr_cd],
[business_area_fact] as [bsn_ar_fct_cd],
substring(gl_account, 1, 4) as [gl_acct_nr],
[project_id] as [proj_id],
[wbs_id] as [wbs_cd],
[po_requestor] as [rqstr_login_id],
coalesce(po_item_local_amount, 0) as [ln_itm_totl_lcl_curr_amt],
coalesce(po_item_usd_amount, 0) as [ln_itm_totl_usd_amt],
coalesce(local_currency, 0) as [lcl_curr_cd],
coalesce(purchase_order_qty, 0) as [qty],
[hours] as [hrs_qty],
[unit_of_measure] as [of_unt_msr_cd],
[payment_terms] as [pmnt_trm_cd],
[requirement_tracking_number] as [rqmt_trkg_id],
[contract_number] as [cctr_id],
[gl_account] as [cctr_nm],
[vendor_material_number] as [mfrr_prt_nr],
[delivery_country] as [ctry_cd],
[ccow_fte] as [ccow_fte_nr],
convert(datetime2(0), po_delivery_date) as [po_dlvry_dt],
[sl_order_code] as [sl_ord_cd],
[sl_labor_key] as [lr_ky],
'Non core skill (commodity skill,100% outsourced)' as [rtnl_for_otsrc_lr_txt],
[po_version] as [po_vrsn_nr],
[po_version_completion_flag] as [po_vrsn_cmpltn_flg],
[ariba_id] as [ariba_id],
case when ariba_id like '%-%' then SUBSTRING(ariba_id, 0, charindex('-',ariba_id))
	 else ariba_id
end as [pr_nr],
case when ariba_id like '%-%' then SUBSTRING(ariba_id, charindex('-', ariba_id)+1, len(ariba_id))
	 else '0'
end as [pr_ln],
'APJ' as [rgn_typ], 
[edl_crt_ts] as [edl_crt_ts],
[edl_upd_ts] as [edl_upd_ts],
[edl_src_sys_ky] as [edl_src_sys_ky],
[edl_upd_tm_ky] as [edl_upd_tm_ky],
[fl_nm] as [fl_nm],
[edl_chcksm_tx] as [edl_chcksm_tx],
[edl_job_sqn_nr] as [edl_job_sqn_nr]
FROM imt_compass_po_apj  
UNION
SELECT
concat_ws('~',purchase_order_number,po_line_number,'COMPASS') as [ky_id],
'COMPASS' as [src_sys_nm],
[purchase_order_number] as [po_nr],
[purchase_order_type] as [po_typ_cd],
[porg] as [porg_cd],
[pgroup] as [prchg_grp_cd],
convert(datetime2(0), po_document_date) as [po_dcmt_dt],
[company_code] as [co_cd],
[vendor_num] as [vndr_nr],
[vendor_name] as [vndr_nm],
[po_line_number] as [po_ln_itm_nr],
[deletion_flag] as [dltn_flg],
[material_group] as [mtrl_grp_cd],
[material_group_description] as [mtrl_grp_dn],
convert(datetime2(0), po_last_modified_date) as [po_last_mfyd_dt],
[material_number] as [mtrl_nr],
[po_line_item_description] as [itm_dn],
[po_plant] as [po_plnt_cd],
[purchase_requisition_number] as [rqstn_nr],
[pr_line_item_number] as [ln_itm_on_req_nr],
[cost_object] as [fncl_mppg_cd],
[profit_center] as [pft_cntr_cd],
[business_area_fact] as [bsn_ar_fct_cd],
substring(gl_account, 1, 4) as [gl_acct_nr],
[project_id] as [proj_id],
[wbs_id] as [wbs_cd],
[po_requestor] as [rqstr_login_id],
coalesce(po_item_local_amount, 0) as [ln_itm_totl_lcl_curr_amt],
coalesce(po_item_usd_amount, 0) as [ln_itm_totl_usd_amt],
coalesce(local_currency, 0) as [lcl_curr_cd],
coalesce(purchase_order_qty, 0) as [qty],
[hours] as [hrs_qty],
[unit_of_measure] as [of_unt_msr_cd],
[payment_terms] as [pmnt_trm_cd],
[requirement_tracking_number] as [rqmt_trkg_id],
[contract_number] as [cctr_id],
[gl_account] as [cctr_nm],
[vendor_material_number] as [mfrr_prt_nr],
[delivery_country] as [ctry_cd],
[ccow_fte] as [ccow_fte_nr],
convert(datetime2(0), po_delivery_date) as [po_dlvry_dt],
[sl_order_code] as [sl_ord_cd],
[sl_labor_key] as [lr_ky],
'Non core skill (commodity skill,100% outsourced)' as [rtnl_for_otsrc_lr_txt],
[po_version] as [po_vrsn_nr],
[po_version_completion_flag] as [po_vrsn_cmpltn_flg],
[ariba_id] as [ariba_id],
case when ariba_id like '%-%' then SUBSTRING(ariba_id, 0, charindex('-',ariba_id))
	 else ariba_id
end as [pr_nr],
case when ariba_id like '%-%' then SUBSTRING(ariba_id, charindex('-', ariba_id)+1, len(ariba_id))
	 else '0'
end as [pr_ln],
'EMEA' as [rgn_typ], 
[edl_crt_ts] as [edl_crt_ts],
[edl_upd_ts] as [edl_upd_ts],
[edl_src_sys_ky] as [edl_src_sys_ky],
[edl_upd_tm_ky] as [edl_upd_tm_ky],
[fl_nm] as [fl_nm],
[edl_chcksm_tx] as [edl_chcksm_tx],
[edl_job_sqn_nr] as [edl_job_sqn_nr]
FROM imt_compass_po_emea
)
INSERT INTO [scm_ing].[ing_cmpss_po]
SELECT * FROM CTE_UNION
WHERE po_nr NOT LIKE 'X%';
END
GO
```