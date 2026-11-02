-- bronze table: products
-- full extract; incremental loads filter on the source's updated_at watermark
SELECT *
FROM catalog.products;
