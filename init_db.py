import sqlite3
import os

def init_db():
    db_path = os.path.join('database', 'isl.db')
    
    # Ensure the database directory exists
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create Gestures Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS gestures (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            word TEXT NOT NULL,
            gesture_file TEXT NOT NULL
        )
    ''');
    
    # Add sample data if table is empty
    cursor.execute('SELECT COUNT(*) FROM gestures')
    if cursor.fetchone()[0] == 0:
        samples = [
            ('hello', 'hello.gif'),
            ('thanks', 'thanks.gif'),
            ('yes', 'yes.gif'),
            ('no', 'no.gif')
        ]
        cursor.executemany('INSERT INTO gestures (word, gesture_file) VALUES (?, ?)', samples)
        print("Sample data inserted into database.")

    
    conn.commit()
    conn.close()
    print(f"Database initialized at {db_path}")

if __name__ == '__main__':
    init_db()
