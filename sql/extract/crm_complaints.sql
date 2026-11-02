-- bronze table: crm_complaints
-- full extract; incremental loads filter on the source's updated_at watermark
SELECT *
FROM crm.complaints;
