import os
import subprocess
import sqlite3

dataset_dir = "dataset"
output_dir = "static/gestures"
db_path = "database/isl.db"

os.makedirs(output_dir, exist_ok=True)

def get_db():
    return sqlite3.connect(db_path)

folders = [f for f in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, f))]

conn = get_db()
cur = conn.cursor()

# Ensure the table exists
cur.execute("CREATE TABLE IF NOT EXISTS gestures (id INTEGER PRIMARY KEY, word TEXT, gesture_file TEXT)")

for folder in folders:
    class_dir = os.path.join(dataset_dir, folder)
    # Find the first .MOV file
    videos = [f for f in os.listdir(class_dir) if f.lower().endswith(".mov")]
    if not videos:
        continue
        
    input_video = os.path.join(class_dir, videos[0])
    output_video_name = f"{folder}.mp4"
    output_video_path = os.path.join(output_dir, output_video_name)
    
    print(f"[INFO] Converting {input_video} to {output_video_path}...")
    
    # Convert using ffmpeg
    try:
        subprocess.run([
            "ffmpeg", "-y", "-i", input_video, 
            "-vcodec", "libx264", "-crf", "25", "-pix_fmt", "yuv420p",
            output_video_path
        ], check=True, capture_output=True)
        
        # Update database
        # Check if word exists
        cur.execute("SELECT id FROM gestures WHERE word=?", (folder,))
        row = cur.fetchone()
        if row:
            cur.execute("UPDATE gestures SET gesture_file=? WHERE word=?", (output_video_name, folder))
        else:
            cur.execute("INSERT INTO gestures (word, gesture_file) VALUES (?, ?)", (folder, output_video_name))
        
        print(f"[SUCCESS] Processed {folder}")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to convert {folder}: {e.stderr.decode()}")

conn.commit()
conn.close()
print("[INFO] Done processing videos.")
