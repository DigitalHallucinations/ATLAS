
import hashlib
import hmac
import os
import sqlite3
from ATLAS.config import ConfigManager
from modules.logging.logger import setup_logger


class UserAccountDatabase:
    def __init__(self, db_name="User.db", base_dir=None):
        self.config_manager = ConfigManager
        self.logger = setup_logger(__name__)

        if base_dir is not None:
            root_dir = os.path.abspath(base_dir)
        else:
            root_dir = os.path.dirname(os.path.abspath(__file__))
        user_profiles_dir = os.path.join(root_dir, 'user_profiles')

        if not os.path.exists(user_profiles_dir):
            os.makedirs(user_profiles_dir)

        db_path = os.path.join(user_profiles_dir, db_name)

        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()

        self.create_table()

    def create_table(self):
        query = """
        CREATE TABLE IF NOT EXISTS user_accounts (
            id INTEGER PRIMARY KEY,
            username TEXT NOT NULL,
            password TEXT NOT NULL,
            email TEXT NOT NULL,
            name TEXT,
            DOB TEXT
        );
        """
        self.cursor.execute(query)
        self.conn.commit()

    @staticmethod
    def _hash_password(password, *, iterations=100_000):
        if password is None:
            raise ValueError("Password must not be None when hashing.")

        salt = os.urandom(16)
        password_bytes = password.encode('utf-8')
        dk = hashlib.pbkdf2_hmac('sha256', password_bytes, salt, iterations)
        return f"pbkdf2_sha256${iterations}${salt.hex()}${dk.hex()}"

    @staticmethod
    def _verify_password(stored_hash, candidate_password):
        if not stored_hash or candidate_password is None:
            return False

        try:
            algorithm, iterations, salt_hex, hash_hex = stored_hash.split('$', 3)
            iterations = int(iterations)
            salt = bytes.fromhex(salt_hex)
            expected_hash = bytes.fromhex(hash_hex)
        except (ValueError, TypeError):
            return False

        if algorithm != 'pbkdf2_sha256':
            return False

        candidate_bytes = candidate_password.encode('utf-8')
        candidate_hash = hashlib.pbkdf2_hmac('sha256', candidate_bytes, salt, iterations)
        return hmac.compare_digest(candidate_hash, expected_hash)

    def add_user(self, username, password, email, name, dob):
        hashed_password = self._hash_password(password)
        query = """
        INSERT INTO user_accounts (username, password, email, name, DOB)
        VALUES (?, ?, ?, ?, ?)
        """
        self.cursor.execute(query, (username, hashed_password, email, name, dob))
        self.conn.commit()

    def get_user(self, username):
        query = "SELECT * FROM user_accounts WHERE username = ?"
        self.cursor.execute(query, (username,))
        return self.cursor.fetchone()

    def get_all_users(self):
        query = "SELECT * FROM user_accounts"
        self.cursor.execute(query)
        return self.cursor.fetchall()
    
    def get_user_profile(self, username):
        query = "SELECT * FROM user_accounts WHERE username = ?"
        self.cursor.execute(query, (username,))
        result = self.cursor.fetchone()
        
        return result

    def close_connection(self):
        """Close the connection to the SQLite database."""
        self.logger.info("Closing UA database connection.")
        try:
            self.conn.close()
        except sqlite3.Error as e:
            self.logger.info(f"Error closing connection: {e}")
            raise 

    def update_user(self, username, password=None, email=None, name=None, dob=None):
        if password:
            hashed_password = self._hash_password(password)
            query = "UPDATE user_accounts SET password = ? WHERE username = ?"
            self.cursor.execute(query, (hashed_password, username))
        if email:
            query = "UPDATE user_accounts SET email = ? WHERE username = ?"
            self.cursor.execute(query, (email, username))
        if name:
            query = "UPDATE user_accounts SET name = ? WHERE username = ?"
            self.cursor.execute(query, (name, username))
        if dob:
            query = "UPDATE user_accounts SET DOB = ? WHERE username = ?"
            self.cursor.execute(query, (dob, username))
        self.conn.commit()

    def verify_user_password(self, username, candidate_password):
        user_record = self.get_user(username)
        if not user_record:
            return False

        stored_hash = user_record[2]
        return self._verify_password(stored_hash, candidate_password)
