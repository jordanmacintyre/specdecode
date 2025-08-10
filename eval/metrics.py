import json
import os
from datetime import datetime
from pprint import pprint


class DecodeMetrics:
    def __init__(self, config):
        self.reset(config=config)

    def reset(self, config):
        self.summary_file_name = config["summary_file_name"]
        self.max_new_tokens = config["max_new_tokens"]
        self.k = config["k"]
        self.calls = 0
        self.total_time = 0.0
        self.total_accepted_tokens = 0
        self.total_tokens = 0
        self.baseline_time = None

    def record_total_time(self, t):
        self.total_time = t

    def set_baseline_time(self, t):
        self.baseline_time = t

    def add_tokens(self, total=0, accepted=0, calls=0):
        self.total_accepted_tokens += accepted
        self.total_tokens += total
        self.calls += calls

    def acceptance_rate(self):
        total_possible = self.calls * self.k
        return (
            self.total_accepted_tokens / total_possible if total_possible > 0 else 0.0
        )

    def throughput_total(self):
        return self.total_tokens / self.total_time if self.total_time > 0 else 0.0

    def throughput_baseline(self):
        return self.max_new_tokens / self.baseline_time if self.baseline_time else 0.0

    def avg_latency_per_token(self):
        return self.total_time / self.total_tokens if self.total_tokens > 0 else 0.0

    def speedup(self):
        if self.baseline_time and self.total_time:
            return self.baseline_time / self.total_time
        return None

    def summary(self):
        return {
            "tokens_generated": self.total_tokens,
            "accepted_tokens": self.total_accepted_tokens,
            "total_time": round(self.total_time, 4),
            "baseline_time": round(self.baseline_time, 4),
            "avg_latency_per_token (s)": round(self.avg_latency_per_token(), 4),
            "acceptance_rate": round(self.acceptance_rate(), 4),
            "tokens/s (speculative pipeline)": round(self.throughput_total(), 2),
            "tokens/s (baseline)": round(self.throughput_baseline(), 2),
            "speedup": round(self.speedup(), 2) if self.speedup() else None,
        }

    def print_summary(self):
        print("\nDecode Metrics Summary")
        print("--------------------------")
        pprint(self.summary(), sort_dicts=False)

    def save_summary(self, directory="results"):
        os.makedirs(directory, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(
            directory, f"metrics_{self.summary_file_name}_{timestamp}.json"
        )

        with open(filepath, "w") as f:
            json.dump(self.summary(), f, indent=2)

        print(f"\nMetrics saved to {filepath}")
