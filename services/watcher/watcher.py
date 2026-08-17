import time
from pathlib import Path
from watchdog.observers.polling import PollingObserver as Observer
from watchdog.events import FileSystemEventHandler
import pandas as pd

EXPORT_DIR = Path("/mnt/c/Users/janni/Documents/Sports Interactive/Football Manager 26/FM26PlayerExport by vinteset/Exports CSV")

def wait_until_stable(path, checks=10, interval=0.5):
    last_size = -1
    for _ in range(checks):
        if not path.exists():
            return False
        size = path.stat().st_size
        if size > 0 and size == last_size:
            return True
        last_size = size
        time.sleep(interval)
    return False

class ExportHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory:
            return
        path = Path(event.src_path)
        if path.suffix.lower() != ".csv":
            return

        print(f"Ny fil opdaget: {path.name}")

        if not wait_until_stable(path):
            print(f"Filen blev aldrig stabil: {path.name}")
            return

        try:
            df = pd.read_csv(path, sep=";", encoding="utf-8")
        except Exception as e:
            print(f"Kunne ikke læse filen endnu: {e}")
            return

        print(f"Parsed {len(df)} spillere, {len(df.columns)} kolonner")
        print(df[["Spiller", "Position", "Alder"]].head())

if __name__ == "__main__":
    observer = Observer()
    observer.schedule(ExportHandler(), str(EXPORT_DIR), recursive=False)
    observer.start()
    print(f"Overvåger: {EXPORT_DIR}")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
