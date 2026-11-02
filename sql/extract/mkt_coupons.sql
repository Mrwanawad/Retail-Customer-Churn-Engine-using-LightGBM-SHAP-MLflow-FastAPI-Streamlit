-- bronze table: mkt_coupons
-- full extract; incremental loads filter on the source's updated_at watermark
SELECT *
FROM marketing.coupons;
