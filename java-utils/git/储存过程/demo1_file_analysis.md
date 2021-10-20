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



#### sql

首先这个sql文件里面的代码是一个存储过程。这个存储过程的作用是将联合和转换的数据插入目标表。

首先使用了关键字 with as ：定义了一个SQL片段，该SQL片段会被整个SQL语句所用到，文件中sql片段里面包含了**数据的选择、转换和连接**，也就是包含三个select语句的连接。

- 首先是将数据进行转换和连接。转换函数使用的是convert函数。这个函数的作用是在不同字符集之间转换数据，根据存储过程中的convert函数的语法，CONVERT(expr, type)语法包含一个表达式和一个类型值（用来指定结果的类型），生成指定类型的结果值。
- 这里使用了UNION将多个SELECT语句的结果合并到一个结果集中。使用UNION关键字需要注意UNION 内部的 SELECT 语句必须拥有相同数量的列。
- 用到了case when表达式。

sql文件的最后将插入到ING表中的数据查询了出来。应该用来查看插入到目标表中的数据。

#### get_pklist_and_col_mapping

首先这个函数的作用应该是 选择出提供数据的表（从这个表里面选择数据并转换），选择的列表达式（定义选择出来的数据列名），和目标表（将数据插入到这个表里面），它是通过match方法确定的。

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