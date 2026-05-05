-- Create user_auth table for database-based authentication
CREATE TABLE IF NOT EXISTS user_auth (
    authid SERIAL PRIMARY KEY,
    userid INTEGER NOT NULL UNIQUE,
    username VARCHAR(50) NOT NULL UNIQUE,
    hashed_password VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_userid FOREIGN KEY (userid) REFERENCES users(userid) ON DELETE CASCADE
);

-- Create indexes for faster lookups
CREATE INDEX idx_user_auth_username ON user_auth(username);
CREATE INDEX idx_user_auth_userid ON user_auth(userid);
