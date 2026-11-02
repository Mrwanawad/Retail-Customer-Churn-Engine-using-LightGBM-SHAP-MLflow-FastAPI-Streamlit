-- bronze table: pos_v2
-- full extract; incremental loads filter on the source's updated_at watermark
SELECT *
FROM pos.sales_lines;
