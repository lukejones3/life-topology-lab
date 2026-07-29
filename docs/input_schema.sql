PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

CREATE TABLE IF NOT EXISTS pipeline_runs (
  run_id TEXT PRIMARY KEY,
  pipeline_version TEXT NOT NULL,
  started_at TEXT NOT NULL,
  completed_at TEXT,
  status TEXT NOT NULL,
  manifest_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS atomic_units (
  unit_id TEXT PRIMARY KEY,
  source_name TEXT NOT NULL,
  source_row_id TEXT NOT NULL,
  recorded_at TEXT NOT NULL,
  civil_day TEXT NOT NULL,
  behavioral_day TEXT NOT NULL,
  channel TEXT NOT NULL,
  direction TEXT,
  thread TEXT,
  person_id TEXT,
  kind TEXT,
  text TEXT NOT NULL,
  n_chars INTEGER NOT NULL,
  origin_class TEXT NOT NULL,
  origin_confidence REAL NOT NULL,
  temporal_mode TEXT NOT NULL,
  temporal_confidence REAL NOT NULL,
  technical_share REAL NOT NULL DEFAULT 0,
  copied_share REAL NOT NULL DEFAULT 0,
  metadata_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_atomic_behavioral_day ON atomic_units(behavioral_day);
CREATE INDEX IF NOT EXISTS idx_atomic_thread_time ON atomic_units(channel, thread, recorded_at);
CREATE INDEX IF NOT EXISTS idx_atomic_origin ON atomic_units(origin_class);

CREATE TABLE IF NOT EXISTS episodes (
  episode_id TEXT PRIMARY KEY,
  behavioral_day TEXT NOT NULL,
  started_at TEXT NOT NULL,
  ended_at TEXT NOT NULL,
  channel TEXT NOT NULL,
  thread TEXT,
  person_id TEXT,
  temporal_mode TEXT NOT NULL,
  origin_class TEXT NOT NULL,
  unit_count INTEGER NOT NULL,
  char_count INTEGER NOT NULL,
  text TEXT NOT NULL,
  evidence_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_episode_day ON episodes(behavioral_day);

CREATE TABLE IF NOT EXISTS temporal_references (
  reference_id TEXT PRIMARY KEY,
  source_unit_id TEXT NOT NULL,
  recorded_at TEXT NOT NULL,
  interpretation_as_of TEXT NOT NULL,
  relation_type TEXT NOT NULL,
  target_start TEXT,
  target_end TEXT,
  target_text TEXT,
  target_entity TEXT,
  confidence REAL NOT NULL,
  extractor TEXT NOT NULL,
  evidence_text TEXT NOT NULL,
  FOREIGN KEY(source_unit_id) REFERENCES atomic_units(unit_id)
);
CREATE INDEX IF NOT EXISTS idx_temporal_source ON temporal_references(source_unit_id);
CREATE INDEX IF NOT EXISTS idx_temporal_target ON temporal_references(target_start, target_end);

CREATE TABLE IF NOT EXISTS semantic_links (
  link_id TEXT PRIMARY KEY,
  source_item_id TEXT NOT NULL,
  target_item_id TEXT NOT NULL,
  source_day TEXT NOT NULL,
  target_day TEXT NOT NULL,
  relation_type TEXT NOT NULL,
  similarity REAL NOT NULL,
  elapsed_days INTEGER NOT NULL,
  confidence REAL NOT NULL,
  method TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_semantic_link_source ON semantic_links(source_item_id);
CREATE INDEX IF NOT EXISTS idx_semantic_link_target ON semantic_links(target_item_id);

CREATE TABLE IF NOT EXISTS day_facets (
  behavioral_day TEXT NOT NULL,
  facet TEXT NOT NULL,
  view_type TEXT NOT NULL,
  text TEXT NOT NULL,
  unit_count INTEGER NOT NULL,
  char_count INTEGER NOT NULL,
  evidence_json TEXT NOT NULL,
  PRIMARY KEY(behavioral_day, facet, view_type)
);

CREATE TABLE IF NOT EXISTS daily_features (
  behavioral_day TEXT PRIMARY KEY,
  civil_date TEXT NOT NULL,
  event_count INTEGER NOT NULL,
  user_authored_count INTEGER NOT NULL,
  original_reflection_count INTEGER NOT NULL,
  question_count INTEGER NOT NULL,
  technical_count INTEGER NOT NULL,
  copied_count INTEGER NOT NULL,
  social_count INTEGER NOT NULL,
  media_count INTEGER NOT NULL,
  distinct_people INTEGER NOT NULL,
  distinct_threads INTEGER NOT NULL,
  active_minutes REAL NOT NULL,
  late_night_count INTEGER NOT NULL,
  chars_total INTEGER NOT NULL,
  chars_original INTEGER NOT NULL,
  reflection_references INTEGER NOT NULL,
  retrospective_count INTEGER NOT NULL,
  prospective_count INTEGER NOT NULL,
  chapter_count INTEGER NOT NULL DEFAULT 0,
  metadata_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS daily_sensors (
  behavioral_day TEXT PRIMARY KEY,
  music_plays INTEGER NOT NULL DEFAULT 0,
  music_minutes REAL NOT NULL DEFAULT 0,
  music_repeat_ratio REAL NOT NULL DEFAULT 0,
  top_songs_json TEXT NOT NULL,
  youtube_events INTEGER NOT NULL DEFAULT 0,
  youtube_searches INTEGER NOT NULL DEFAULT 0,
  browser_events INTEGER NOT NULL DEFAULT 0,
  web_searches INTEGER NOT NULL DEFAULT 0,
  sensor_text TEXT NOT NULL,
  evidence_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS daily_impact (
  behavioral_day TEXT PRIMARY KEY,
  activity_z REAL NOT NULL,
  novelty REAL NOT NULL,
  lived_shift REAL NOT NULL,
  interpretive_shift REAL NOT NULL,
  prospective_shift REAL NOT NULL,
  persistence REAL NOT NULL,
  cross_domain_spread REAL NOT NULL,
  event_impact REAL NOT NULL,
  interpretive_impact REAL NOT NULL,
  activation_intensity REAL NOT NULL,
  residual_impact REAL NOT NULL,
  evidence_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS embeddings (
  item_id TEXT NOT NULL,
  item_type TEXT NOT NULL,
  content_hash TEXT NOT NULL,
  model_name TEXT NOT NULL,
  dimensions INTEGER NOT NULL,
  embedded_at TEXT NOT NULL,
  PRIMARY KEY(item_id, item_type, model_name)
);
