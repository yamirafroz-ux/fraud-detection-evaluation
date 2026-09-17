-- Run with: sqlite3 -header -column runs/demo/transactions.sqlite < sql/analysis.sql
-- Ground-truth labels are used for retrospective analysis ONLY.
SELECT CAST(time / 24 AS INTEGER) AS day,
       COUNT(*) AS attempts,
       ROUND(AVG(is_fraud), 4) AS fraud_prevalence,
       SUM(CASE WHEN accepted = 1 AND is_fraud = 1 THEN amount_cents ELSE 0 END) / 100.0 AS settled_fraud_value
FROM transactions GROUP BY day ORDER BY day;

-- Incoming concentration over the TRAINING period only (default 45-day horizon).
SELECT receiver, COUNT(DISTINCT sender) AS unique_senders, COUNT(*) AS incoming_attempts
FROM transactions WHERE time < 540
GROUP BY receiver ORDER BY unique_senders DESC LIMIT 10;

-- Exact causal window matching Python: (t-24, t), excluding current event.
SELECT t.transaction_id, t.sender, t.time,
       (SELECT COUNT(*) FROM transactions h
        WHERE h.sender = t.sender AND h.time > t.time - 24 AND h.time < t.time
       ) AS previous_24h,
       t.time - LAG(t.time) OVER (PARTITION BY t.sender ORDER BY t.time) AS hours_since_last
FROM transactions t ORDER BY t.time LIMIT 30;
