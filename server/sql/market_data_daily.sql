CREATE EXTENSION IF NOT EXISTS timescaledb;

CREATE TABLE IF NOT EXISTS market_data_daily (
    date        TIMESTAMPTZ       NOT NULL,
    symbol      TEXT              NOT NULL,
    open        DOUBLE PRECISION  NOT NULL,
    high        DOUBLE PRECISION  NOT NULL,
    low         DOUBLE PRECISION  NOT NULL,
    close       DOUBLE PRECISION  NOT NULL,
    volume      DOUBLE PRECISION  NOT NULL,
    amount      DOUBLE PRECISION  NOT NULL DEFAULT 0,
    turn        DOUBLE PRECISION  NOT NULL DEFAULT 0,
    tradestatus INT               NOT NULL DEFAULT 1,
    is_st       INT               NOT NULL DEFAULT 0,
    PRIMARY KEY (symbol, date)
);

SELECT create_hypertable('market_data_daily', 'date', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_symbol_date ON market_data_daily (symbol, date DESC);

ALTER TABLE market_data_daily SET (
  timescaledb.compress,
  timescaledb.compress_segmentby = 'symbol'
);

SELECT add_compression_policy('market_data_daily', INTERVAL '7 days');

-- Migration: add new columns to existing table (safe to run repeatedly)
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='market_data_daily' AND column_name='amount') THEN
        ALTER TABLE market_data_daily ADD COLUMN amount DOUBLE PRECISION NOT NULL DEFAULT 0;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='market_data_daily' AND column_name='turn') THEN
        ALTER TABLE market_data_daily ADD COLUMN turn DOUBLE PRECISION NOT NULL DEFAULT 0;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='market_data_daily' AND column_name='tradestatus') THEN
        ALTER TABLE market_data_daily ADD COLUMN tradestatus INT NOT NULL DEFAULT 1;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='market_data_daily' AND column_name='is_st') THEN
        ALTER TABLE market_data_daily ADD COLUMN is_st INT NOT NULL DEFAULT 0;
    END IF;
END $$;
