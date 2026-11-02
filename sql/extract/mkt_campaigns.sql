-- bronze table: mkt_campaigns
-- full extract; incremental loads filter on the source's updated_at watermark
SELECT *
FROM marketing.campaigns;
