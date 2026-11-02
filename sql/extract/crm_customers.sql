-- bronze table: crm_customers
-- full extract; incremental loads filter on the source's updated_at watermark
SELECT *
FROM crm.customers;
