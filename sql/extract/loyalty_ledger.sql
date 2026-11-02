-- bronze table: loyalty_ledger
-- full extract; incremental loads filter on the source's updated_at watermark
SELECT *
FROM loyalty.points_ledger;
