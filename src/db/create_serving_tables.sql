-- Chapter 11: the entity history store.
-- Holds only what the aggregates need, for days 0-154. The test partition is
-- never written here; a protocol check confirms it.
CREATE TABLE IF NOT EXISTS tx_history (
    transactionid  BIGINT PRIMARY KEY,
    transactiondt  BIGINT NOT NULL,
    transactionamt DOUBLE PRECISION,
    uid_card       TEXT NOT NULL,
    uid_card_addr  TEXT NOT NULL,
    uid_account    TEXT NOT NULL,
    productcd_key  TEXT NOT NULL,
    devicetype_key TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_tx_history_account  ON tx_history (uid_account, transactiondt);
CREATE INDEX IF NOT EXISTS ix_tx_history_cardaddr ON tx_history (uid_card_addr, transactiondt);
CREATE INDEX IF NOT EXISTS ix_tx_history_card     ON tx_history (uid_card, transactiondt);

-- Every decision the system makes. Chapter 12 reads this rather than a
-- remembered reference: scored_at for drift windows, probability for score
-- drift, reason_codes for attribution drift.
CREATE TABLE IF NOT EXISTS predictions (
    id              BIGSERIAL PRIMARY KEY,
    transactionid   BIGINT NOT NULL,
    scored_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    model_name      TEXT NOT NULL,
    model_version   TEXT NOT NULL,
    probability_raw DOUBLE PRECISION NOT NULL,
    probability     DOUBLE PRECISION NOT NULL,
    threshold       DOUBLE PRECISION NOT NULL,
    decision        TEXT NOT NULL,
    reason_codes    JSONB,
    features        JSONB,
    latency_ms      DOUBLE PRECISION
);

CREATE INDEX IF NOT EXISTS ix_predictions_tid       ON predictions (transactionid);
CREATE INDEX IF NOT EXISTS ix_predictions_scored_at ON predictions (scored_at);