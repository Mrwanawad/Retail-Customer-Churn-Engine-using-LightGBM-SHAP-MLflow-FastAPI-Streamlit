-- bronze table: mkt_targets
-- full extract; incremental loads filter on the source's updated_at watermark
SELECT *
FROM marketing.campaign_targets;
