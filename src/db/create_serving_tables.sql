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