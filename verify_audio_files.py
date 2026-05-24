import os
import pandas as pd
from pathlib import Path

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "processed_labels.csv")
AUDIO_DIR = os.path.join(BASE_DIR, "processed_clips")

def verify_audio_files():
    print("="*60)
    print(" AUDIO FILES VERIFICATION ")
    print("="*60)
    
    # Check if CSV exists
    if not os.path.exists(CSV_PATH):
        print(f"[ERROR] CSV file not found: {CSV_PATH}")
        return
    
    # Check if audio directory exists
    if not os.path.exists(AUDIO_DIR):
        print(f"[ERROR] Audio directory not found: {AUDIO_DIR}")
        return
    
    # Load CSV
    df = pd.read_csv(CSV_PATH)
    print(f"Total clips in CSV: {len(df)}")
    
    # Count audio files
    audio_files = list(Path(AUDIO_DIR).glob("*.wav"))
    print(f"Total WAV files in folder: {len(audio_files)}")
    
    # Check how many CSV entries have matching audio files
    matched = 0
    missing = 0
    
    for idx, row in df.iterrows():
        show = str(row['Show']).strip()
        ep_id = str(row['EpId']).strip()
        clip_id = str(row['ClipId']).strip()
        
        # Handle FluencyBank naming
        if "fluencybank" in show.lower():
            save_ep_id = ep_id.zfill(3)
        else:
            save_ep_id = ep_id
        
        filename = f"{show}_{save_ep_id}_{clip_id}.wav"
        filepath = os.path.join(AUDIO_DIR, filename)
        
        if os.path.exists(filepath):
            matched += 1
        else:
            missing += 1
            if missing <= 10:  # Show first 10 missing files
                print(f"  Missing: {filename}")
    
    print()
    print("="*60)
    print(" VERIFICATION RESULTS ")
    print("="*60)
    print(f"CSV entries with audio: {matched} ({matched/len(df)*100:.1f}%)")
    print(f"CSV entries missing audio: {missing} ({missing/len(df)*100:.1f}%)")
    print("="*60)
    
    # Check class distribution
    print()
    print("CLASS DISTRIBUTION:")
    print(f"Stutter: {df['is_stutter'].sum()} ({df['is_stutter'].sum()/len(df)*100:.1f}%)")
    print(f"Fluent: {(~df['is_stutter'].astype(bool)).sum()} ({(~df['is_stutter'].astype(bool)).sum()/len(df)*100:.1f}%)")
    print("="*60)

if __name__ == "__main__":
    verify_audio_files()
