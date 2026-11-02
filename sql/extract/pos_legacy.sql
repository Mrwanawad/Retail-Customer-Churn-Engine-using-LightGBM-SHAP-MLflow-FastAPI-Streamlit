-- bronze table: pos_legacy
-- full extract; incremental loads filter on the source's updated_at watermark
SELECT *
FROM pos.sales_lines_legacy;
