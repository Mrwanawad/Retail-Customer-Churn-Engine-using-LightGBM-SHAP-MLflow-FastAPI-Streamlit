-- bronze table: mkt_redemptions
-- full extract; incremental loads filter on the source's updated_at watermark
SELECT *
FROM marketing.coupon_redemptions;
