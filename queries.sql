-- =====================================================================
-- RetainIQ — SQL queries for subscription churn & LTV analysis
-- =====================================================================
-- Same analysis as build_project.py, written against a realistic
-- data-warehouse schema (PostgreSQL / Redshift dialect).
--
-- Hypothetical schema (typical Kimball-style for a subscription business):
--
--   dim_subscribers          : one row per subscriber, dimensional attributes
--   fact_subscriber_events   : one row per event (NPS, payment, swap, etc.)
--
-- Every query is independently runnable. CTEs are used throughout for
-- readability. Replace date literals with bind parameters in production.
-- =====================================================================


-- ---------------------------------------------------------------------
-- Schema reference (for context)
-- ---------------------------------------------------------------------
-- CREATE TABLE dim_subscribers (
--   subscriber_id        BIGINT PRIMARY KEY,
--   signup_date          DATE NOT NULL,
--   city                 VARCHAR(50),
--   acquisition_channel  VARCHAR(50),       -- google_ads | meta_ads | organic | referral | direct
--   product_category     VARCHAR(50),       -- sofa | bed | wardrobe | dining | study | full_home
--   plan_tenure_months   SMALLINT,          -- 3 | 6 | 12
--   monthly_rental_inr   NUMERIC(10,2),
--   subscriber_age       SMALLINT,
--   cancelled_date       DATE,              -- NULL if still active
--   current_status       VARCHAR(20)        -- 'active' | 'cancelled'
-- );
--
-- CREATE TABLE fact_subscriber_events (
--   event_id             BIGSERIAL PRIMARY KEY,
--   subscriber_id        BIGINT NOT NULL REFERENCES dim_subscribers(subscriber_id),
--   event_type           VARCHAR(30) NOT NULL,
--     -- 'delivery_complete' | 'payment_success' | 'payment_failure'
--     -- 'nps_response'      | 'swap_request'    | 'support_ticket'
--   event_date           DATE NOT NULL,
--   event_numeric_value  NUMERIC(10,2),     -- NPS score, payment amount, days late, etc.
--   event_text_value     VARCHAR(500)
-- );


-- =====================================================================
-- Q1. COHORT RETENTION TABLE
-- =====================================================================
-- For each acquisition cohort (signup month), compute the % of subscribers
-- still active at each subsequent month. Output is long-format — pivot in
-- the BI tool for the classic retention heatmap.
-- =====================================================================
WITH cohort_base AS (
    SELECT
        DATE_TRUNC('month', signup_date)::DATE              AS cohort_month,
        subscriber_id,
        signup_date,
        COALESCE(cancelled_date, CURRENT_DATE)              AS effective_end_date
    FROM dim_subscribers
    WHERE signup_date BETWEEN DATE '2024-01-01' AND DATE '2024-12-31'
),
months_active AS (
    SELECT
        cohort_month,
        subscriber_id,
        (DATE_PART('year',  AGE(effective_end_date, signup_date)) * 12
       + DATE_PART('month', AGE(effective_end_date, signup_date)))::INT  AS months_lived
    FROM cohort_base
),
cohort_sizes AS (
    SELECT cohort_month, COUNT(*) AS cohort_size
    FROM cohort_base
    GROUP BY cohort_month
),
retention_long AS (
    SELECT
        ma.cohort_month,
        m.month_index,
        SUM(CASE WHEN ma.months_lived >= m.month_index THEN 1 ELSE 0 END) AS retained
    FROM months_active ma
    CROSS JOIN generate_series(0, 18) AS m(month_index)
    GROUP BY ma.cohort_month, m.month_index
)
SELECT
    r.cohort_month,
    r.month_index                                       AS months_since_signup,
    r.retained,
    cs.cohort_size,
    ROUND(100.0 * r.retained / cs.cohort_size, 1)       AS retention_pct
FROM retention_long r
JOIN cohort_sizes  cs ON cs.cohort_month = r.cohort_month
WHERE r.month_index <= EXTRACT(MONTH FROM AGE(CURRENT_DATE, r.cohort_month))
ORDER BY r.cohort_month, r.month_index;


-- =====================================================================
-- Q2. 12-MONTH LTV (BLENDED + ARPU)
-- =====================================================================
-- LTV(12) = sum over months 1..12 of (retention_rate * ARPU).
-- =====================================================================
WITH monthly_retention AS (
    SELECT
        m.month_index,
        AVG(CASE WHEN months_lived >= m.month_index THEN 1.0 ELSE 0.0 END) AS retention_rate
    FROM (
        SELECT
            subscriber_id,
            (DATE_PART('year',  AGE(COALESCE(cancelled_date, CURRENT_DATE), signup_date)) * 12
           + DATE_PART('month', AGE(COALESCE(cancelled_date, CURRENT_DATE), signup_date)))::INT AS months_lived
        FROM dim_subscribers
        WHERE signup_date <= DATE '2024-12-31' - INTERVAL '12 months'
    ) ml
    CROSS JOIN generate_series(0, 12) AS m(month_index)
    GROUP BY m.month_index
),
arpu AS (
    SELECT AVG(monthly_rental_inr) AS arpu_inr FROM dim_subscribers
)
SELECT
    SUM(mr.retention_rate * a.arpu_inr)::INT          AS ltv_12_month_inr,
    MAX(a.arpu_inr)::INT                              AS arpu_inr
FROM monthly_retention mr
CROSS JOIN arpu a;


-- =====================================================================
-- Q3. LTV BY ACQUISITION CHANNEL (12-MONTH HORIZON)
-- =====================================================================
-- Identifies which channels bring the highest-LTV subscribers.
-- Output ranked, suitable for a marketing-mix reallocation decision.
-- =====================================================================
WITH subscriber_tenure AS (
    SELECT
        acquisition_channel,
        monthly_rental_inr,
        (DATE_PART('year',  AGE(COALESCE(cancelled_date, CURRENT_DATE), signup_date)) * 12
       + DATE_PART('month', AGE(COALESCE(cancelled_date, CURRENT_DATE), signup_date)))::INT AS months_lived
    FROM dim_subscribers
    WHERE signup_date <= DATE '2024-12-31' - INTERVAL '12 months'
),
channel_retention AS (
    SELECT
        acquisition_channel,
        m.month_index,
        AVG(CASE WHEN months_lived >= m.month_index THEN 1.0 ELSE 0.0 END) AS retention_rate,
        AVG(monthly_rental_inr)                                            AS channel_arpu
    FROM subscriber_tenure
    CROSS JOIN generate_series(0, 12) AS m(month_index)
    GROUP BY acquisition_channel, m.month_index
)
SELECT
    acquisition_channel,
    SUM(retention_rate * channel_arpu)::INT          AS ltv_12_month_inr,
    AVG(channel_arpu)::INT                           AS arpu_inr
FROM channel_retention
GROUP BY acquisition_channel
ORDER BY ltv_12_month_inr DESC;


-- =====================================================================
-- Q4. AT-RISK SUBSCRIBER SCORING (DAY-30 SNAPSHOT)
-- =====================================================================
-- Builds the feature vector for every subscriber at day 30 — the table the
-- churn classifier consumes. In production this is materialised nightly
-- and fed to the model-scoring pipeline.
-- =====================================================================
WITH eligible_subscribers AS (
    SELECT
        subscriber_id,
        signup_date,
        signup_date + INTERVAL '30 days'    AS snapshot_date,
        city,
        acquisition_channel,
        product_category,
        plan_tenure_months,
        monthly_rental_inr,
        subscriber_age,
        cancelled_date
    FROM dim_subscribers
    WHERE signup_date <= CURRENT_DATE - INTERVAL '30 days'
),
first_delivery AS (
    SELECT
        s.subscriber_id,
        MIN(e.event_date - s.signup_date)::INT       AS first_delivery_delay_days
    FROM eligible_subscribers s
    LEFT JOIN fact_subscriber_events e
      ON e.subscriber_id = s.subscriber_id
     AND e.event_type    = 'delivery_complete'
     AND e.event_date    BETWEEN s.signup_date AND s.snapshot_date
    GROUP BY s.subscriber_id
),
nps_snapshot AS (
    SELECT
        s.subscriber_id,
        AVG(e.event_numeric_value)::NUMERIC(3,1)     AS nps_score
    FROM eligible_subscribers s
    LEFT JOIN fact_subscriber_events e
      ON e.subscriber_id = s.subscriber_id
     AND e.event_type    = 'nps_response'
     AND e.event_date    BETWEEN s.signup_date AND s.snapshot_date
    GROUP BY s.subscriber_id
),
behavioural_counts AS (
    SELECT
        s.subscriber_id,
        COUNT(*) FILTER (WHERE e.event_type = 'payment_failure'
                              AND e.event_date < s.signup_date + INTERVAL '90 days') AS payment_failures_90d,
        COUNT(*) FILTER (WHERE e.event_type = 'swap_request'
                              AND e.event_date < s.signup_date + INTERVAL '60 days') AS swap_requests_60d,
        COUNT(*) FILTER (WHERE e.event_type = 'support_ticket'
                              AND e.event_date < s.signup_date + INTERVAL '60 days') AS support_tickets_60d
    FROM eligible_subscribers s
    LEFT JOIN fact_subscriber_events e ON e.subscriber_id = s.subscriber_id
    GROUP BY s.subscriber_id
)
SELECT
    s.subscriber_id,
    s.snapshot_date,
    s.city,
    s.acquisition_channel,
    s.product_category,
    s.plan_tenure_months,
    s.monthly_rental_inr,
    s.subscriber_age,
    COALESCE(fd.first_delivery_delay_days, 0)        AS first_delivery_delay_days,
    np.nps_score,
    bc.payment_failures_90d,
    bc.swap_requests_60d,
    bc.support_tickets_60d
FROM eligible_subscribers s
LEFT JOIN first_delivery        fd ON fd.subscriber_id = s.subscriber_id
LEFT JOIN nps_snapshot          np ON np.subscriber_id = s.subscriber_id
LEFT JOIN behavioural_counts    bc ON bc.subscriber_id = s.subscriber_id
WHERE s.cancelled_date IS NULL                        -- score active only
ORDER BY s.signup_date DESC;


-- =====================================================================
-- Q5. NPS DETRACTOR RATE BY CHURN MONTH
-- =====================================================================
-- The "3x detractor rate" headline. For each churn month, the share of
-- subscribers who scored NPS <= 6 at their day-30 survey.
-- =====================================================================
WITH first_nps AS (
    SELECT DISTINCT ON (subscriber_id)
        subscriber_id,
        event_numeric_value AS nps_score
    FROM fact_subscriber_events
    WHERE event_type = 'nps_response'
    ORDER BY subscriber_id, event_date ASC
),
churn_classification AS (
    SELECT
        s.subscriber_id,
        CASE
            WHEN s.cancelled_date IS NULL THEN 'retained'
            ELSE 'churned_m' || LEAST(6,
                (DATE_PART('year',  AGE(s.cancelled_date, s.signup_date)) * 12
               + DATE_PART('month', AGE(s.cancelled_date, s.signup_date)))::INT
            )
        END                                          AS churn_bucket,
        n.nps_score
    FROM dim_subscribers s
    LEFT JOIN first_nps n ON n.subscriber_id = s.subscriber_id
    WHERE s.signup_date <= CURRENT_DATE - INTERVAL '120 days'
      AND n.nps_score IS NOT NULL
)
SELECT
    churn_bucket,
    COUNT(*)                                                         AS subscribers,
    AVG(nps_score)::NUMERIC(3,1)                                     AS avg_nps,
    100.0 * SUM(CASE WHEN nps_score <= 6 THEN 1 ELSE 0 END) / COUNT(*)
        AS detractor_pct,
    100.0 * SUM(CASE WHEN nps_score >= 9 THEN 1 ELSE 0 END) / COUNT(*)
        AS promoter_pct
FROM churn_classification
GROUP BY churn_bucket
ORDER BY
    CASE WHEN churn_bucket = 'retained' THEN 99 ELSE 1 END,
    churn_bucket;


-- =====================================================================
-- Q6. STEADY-STATE MONTHLY CHURN (M6 → M12)
-- =====================================================================
-- Excludes the noisy M1–M2 cliff. The assumption input for finance's
-- LTV / unit-economics model.
-- =====================================================================
WITH retention_by_month AS (
    SELECT
        m.month_index,
        AVG(CASE WHEN months_lived >= m.month_index THEN 1.0 ELSE 0.0 END) AS retention_rate
    FROM (
        SELECT
            subscriber_id,
            (DATE_PART('year',  AGE(COALESCE(cancelled_date, CURRENT_DATE), signup_date)) * 12
           + DATE_PART('month', AGE(COALESCE(cancelled_date, CURRENT_DATE), signup_date)))::INT AS months_lived
        FROM dim_subscribers
        WHERE signup_date <= DATE '2024-12-31' - INTERVAL '12 months'
    ) t
    CROSS JOIN generate_series(0, 12) AS m(month_index)
    GROUP BY m.month_index
)
SELECT
    AVG(prev_retention - curr_retention) * 100         AS steady_state_monthly_churn_pct
FROM (
    SELECT
        month_index,
        retention_rate                                                   AS curr_retention,
        LAG(retention_rate) OVER (ORDER BY month_index)                  AS prev_retention
    FROM retention_by_month
    WHERE month_index BETWEEN 6 AND 12
) windowed
WHERE prev_retention IS NOT NULL;


-- =====================================================================
-- Q7. HIGH-RISK SEGMENT ROLLUP (for the retention CRM)
-- =====================================================================
-- Tags every active subscriber with a behavioural segment so the CRM can
-- route the right intervention. Mirrors the segments visible in the BI
-- dashboard's "high-risk segments" panel.
-- =====================================================================
WITH first_nps AS (
    SELECT DISTINCT ON (subscriber_id)
        subscriber_id, event_numeric_value AS nps_score
    FROM fact_subscriber_events
    WHERE event_type = 'nps_response'
    ORDER BY subscriber_id, event_date ASC
),
friction AS (
    SELECT
        s.subscriber_id,
        COUNT(*) FILTER (WHERE e.event_type = 'payment_failure'
                              AND e.event_date < s.signup_date + INTERVAL '90 days') AS payment_failures_90d,
        COUNT(*) FILTER (WHERE e.event_type = 'swap_request'
                              AND e.event_date < s.signup_date + INTERVAL '60 days') AS swap_requests_60d
    FROM dim_subscribers s
    LEFT JOIN fact_subscriber_events e ON e.subscriber_id = s.subscriber_id
    GROUP BY s.subscriber_id
),
labelled AS (
    SELECT
        s.subscriber_id,
        s.acquisition_channel,
        s.monthly_rental_inr,
        s.current_status,
        (DATE_PART('year',  AGE(COALESCE(s.cancelled_date, CURRENT_DATE), s.signup_date)) * 12
       + DATE_PART('month', AGE(COALESCE(s.cancelled_date, CURRENT_DATE), s.signup_date)))::INT AS months_lived,
        n.nps_score,
        f.payment_failures_90d,
        f.swap_requests_60d,
        CASE
            WHEN months_lived >= 12 AND nps_score >= 8 AND COALESCE(payment_failures_90d, 0) = 0
                THEN 'high_value_loyalist'
            WHEN months_lived <= 2 AND (nps_score <= 6 OR payment_failures_90d >= 1 OR swap_requests_60d >= 1)
                THEN 'onboarding_struggler'
            WHEN months_lived BETWEEN 3 AND 6 AND nps_score <= 7
                THEN 'quiet_quitter'
            ELSE 'steady_user'
        END AS segment
    FROM dim_subscribers s
    LEFT JOIN first_nps  n ON n.subscriber_id = s.subscriber_id
    LEFT JOIN friction   f ON f.subscriber_id = s.subscriber_id
)
SELECT
    segment,
    COUNT(*)                                          AS subscribers,
    AVG(months_lived)::NUMERIC(4,1)                   AS avg_months_lived,
    AVG(nps_score)::NUMERIC(3,1)                      AS avg_nps,
    AVG(monthly_rental_inr)::INT                      AS avg_arpu_inr,
    (AVG(months_lived) * AVG(monthly_rental_inr))::INT  AS implied_ltv_inr,
    100.0 * SUM(CASE WHEN current_status = 'cancelled' THEN 1 ELSE 0 END) / COUNT(*)
        AS cancellation_rate_pct
FROM labelled
GROUP BY segment
ORDER BY implied_ltv_inr DESC;


-- =====================================================================
-- End of file.
-- =====================================================================
