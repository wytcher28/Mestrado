import os, json, csv, time

class ExperimentLogger:
    def __init__(self, out_dir: str):
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.round_rows = []
        self.meta = {
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

    def log_meta(self, **kwargs):
        self.meta.update(kwargs)

    def log_round(self, row: dict):
        self.round_rows.append(row)

    def flush(self):
        # meta
        with open(os.path.join(self.out_dir, "meta.json"), "w") as f:
            json.dump(self.meta, f, indent=2)

        # rounds
        if self.round_rows:
            keys = sorted({k for r in self.round_rows for k in r.keys()})
            with open(os.path.join(self.out_dir, "rounds.csv"), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                for r in self.round_rows:
                    w.writerow(r)
