import os
import sys
import logging
import pg8000.dbapi

logging.basicConfig(stream=sys.stderr, level=logging.INFO)
logger = logging.getLogger("init_db")

DB_HOST = os.environ.get("DB_HOST", "ep-withered-breeze-d8845p1k.database.us-east-2.cloud.databricks.com")
DB_NAME = os.environ.get("DB_NAME", "databricks_postgres")
DB_USER = os.environ.get("DB_USER", "research-copilot-agent")
DB_PASS = os.environ.get("DB_PASS", "")

def get_db_conn():
    if not DB_PASS:
        raise ValueError("DB_PASS environment variable is required to connect to Lakebase.")
    return pg8000.dbapi.connect(
        host=DB_HOST, port=5432, database=DB_NAME,
        user=DB_USER, password=DB_PASS, ssl_context=True
    )

TABLES_SQL = [
    """
    CREATE TABLE IF NOT EXISTS users (
        user_id VARCHAR(255) PRIMARY KEY,
        name VARCHAR(255),
        email VARCHAR(255) UNIQUE,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS conversations (
        conversation_id VARCHAR(255) PRIMARY KEY,
        user_id VARCHAR(255) NOT NULL,
        title VARCHAR(255),
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS messages (
        message_id VARCHAR(255) PRIMARY KEY,
        conversation_id VARCHAR(255) NOT NULL REFERENCES conversations(conversation_id) ON DELETE CASCADE,
        role VARCHAR(50) NOT NULL,
        content TEXT NOT NULL,
        citations JSONB DEFAULT '[]'::jsonb,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS learning_goals (
        goal_id VARCHAR(255) PRIMARY KEY,
        user_id VARCHAR(255) NOT NULL,
        title VARCHAR(255) NOT NULL,
        description TEXT,
        target_date DATE,
        is_active BOOLEAN DEFAULT TRUE,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
    );
    """,
    """
    ALTER TABLE learning_goals ADD COLUMN IF NOT EXISTS is_active BOOLEAN DEFAULT TRUE;
    """,
    """
    CREATE TABLE IF NOT EXISTS papers (
        paper_id VARCHAR(255) PRIMARY KEY,
        doi VARCHAR(255),
        title TEXT NOT NULL,
        abstract_text TEXT,
        publication_year INT,
        citation_count INT DEFAULT 0,
        open_access_url TEXT,
        topics JSONB DEFAULT '[]'::jsonb,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS authors (
        author_id VARCHAR(255) PRIMARY KEY,
        display_name VARCHAR(255),
        orcid VARCHAR(255),
        institution_name VARCHAR(255)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS paper_authors (
        paper_id VARCHAR(255) REFERENCES papers(paper_id) ON DELETE CASCADE,
        author_id VARCHAR(255) REFERENCES authors(author_id) ON DELETE CASCADE,
        author_position INT,
        PRIMARY KEY (paper_id, author_id)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS collections (
        collection_id VARCHAR(255) PRIMARY KEY,
        user_id VARCHAR(255) NOT NULL,
        name VARCHAR(255) NOT NULL,
        description TEXT,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS collection_papers (
        collection_id VARCHAR(255) REFERENCES collections(collection_id) ON DELETE CASCADE,
        paper_id VARCHAR(255) REFERENCES papers(paper_id) ON DELETE CASCADE,
        added_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (collection_id, paper_id)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS reading_progress (
        progress_id VARCHAR(255) PRIMARY KEY,
        user_id VARCHAR(255) NOT NULL,
        paper_id VARCHAR(255) REFERENCES papers(paper_id) ON DELETE CASCADE,
        status VARCHAR(50) NOT NULL CHECK (status IN ('TO_READ', 'READING', 'COMPLETED')),
        updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        CONSTRAINT unique_user_paper UNIQUE (user_id, paper_id)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS reading_plans (
        plan_id VARCHAR(255) PRIMARY KEY,
        user_id VARCHAR(255) NOT NULL,
        goal_id VARCHAR(255) REFERENCES learning_goals(goal_id) ON DELETE SET NULL,
        title VARCHAR(255) NOT NULL,
        sequenced_paper_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
        status VARCHAR(50) DEFAULT 'ACTIVE',
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS notes (
        note_id VARCHAR(255) PRIMARY KEY,
        user_id VARCHAR(255) NOT NULL,
        paper_id VARCHAR(255) REFERENCES papers(paper_id) ON DELETE CASCADE,
        content TEXT NOT NULL,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
    );
    """
]

def init_tables():
    try:
        conn = get_db_conn()
        cursor = conn.cursor()
        for query in TABLES_SQL:
            cursor.execute(query)
        conn.commit()
        cursor.close()
        conn.close()
        logger.info("✅ Database tables successfully created/updated.")
    except Exception as e:
        logger.warning("Could not execute table init: %s", e)

if __name__ == "__main__":
    init_tables()
