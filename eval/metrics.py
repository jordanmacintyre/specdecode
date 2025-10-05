import json
import os
from datetime import datetime


class DecodeMetrics:
    def __init__(self, config):
        self.reset(config=config)

    def reset(self, config):
        self.summary_file_name = config.get("summary_file_name", "decode")
        self.batch_size = config.get("batch_size", 1)
        self.num_samples = config.get("num_samples", 1)
        self.k = config["k"]

        # Timing metrics
        self.baseline_time = None
        self.speculative_time = None

        # Token tracking
        self.total_tokens_generated = 0
        self.total_accepted_tokens = 0
        self.num_speculation_rounds = 0

    def set_baseline_time(self, t):
        """Record baseline (target-only) generation time."""
        self.baseline_time = t

    def record_total_time(self, t):
        """Record speculative decoding total time."""
        self.speculative_time = t

    def add_speculation_round(self, num_accepted):
        """
        Record a speculation round.

        Args:
            num_accepted: Number of tokens accepted in this round (0 to k)
        """
        self.num_speculation_rounds += 1
        self.total_accepted_tokens += num_accepted
        self.total_tokens_generated += num_accepted + 1  # accepted + carry token

    def acceptance_rate(self):
        """Percentage of proposed draft tokens that were accepted."""
        total_proposed = self.num_speculation_rounds * self.k
        return (
            self.total_accepted_tokens / total_proposed if total_proposed > 0 else 0.0
        )

    def block_efficiency(self):
        """Average number of tokens accepted per speculation round."""
        return (
            self.total_accepted_tokens / self.num_speculation_rounds
            if self.num_speculation_rounds > 0
            else 0.0
        )

    def throughput_speculative(self):
        """Tokens per second for speculative decoding."""
        return (
            self.total_tokens_generated / self.speculative_time
            if self.speculative_time and self.speculative_time > 0
            else 0.0
        )

    def throughput_baseline(self):
        """Tokens per second for baseline generation."""
        return (
            self.total_tokens_generated / self.baseline_time
            if self.baseline_time and self.baseline_time > 0
            else 0.0
        )

    def latency_per_token_speculative(self):
        """Average time per token for speculative decoding (seconds)."""
        return (
            self.speculative_time / self.total_tokens_generated
            if self.total_tokens_generated > 0
            else 0.0
        )

    def latency_per_token_baseline(self):
        """Average time per token for baseline generation (seconds)."""
        return (
            self.baseline_time / self.total_tokens_generated
            if self.baseline_time and self.total_tokens_generated > 0
            else 0.0
        )

    def speedup(self):
        """Speedup factor vs baseline."""
        if self.baseline_time and self.speculative_time and self.speculative_time > 0:
            return self.baseline_time / self.speculative_time
        return None

    def summary(self):
        """Return summary dictionary of all metrics."""
        return {
            "total_tokens_generated": self.total_tokens_generated,
            "speculation_rounds": self.num_speculation_rounds,
            "acceptance_rate": round(self.acceptance_rate() * 100, 2),  # As percentage
            "block_efficiency": round(self.block_efficiency(), 2),
            "baseline_time_s": (
                round(self.baseline_time, 4) if self.baseline_time else None
            ),
            "speculative_time_s": (
                round(self.speculative_time, 4) if self.speculative_time else None
            ),
            "speedup": round(self.speedup(), 2) if self.speedup() else None,
            "throughput_baseline_tok_per_s": round(self.throughput_baseline(), 2),
            "throughput_speculative_tok_per_s": round(self.throughput_speculative(), 2),
            "latency_per_token_baseline_ms": round(
                self.latency_per_token_baseline() * 1000, 2
            ),
            "latency_per_token_speculative_ms": round(
                self.latency_per_token_speculative() * 1000, 2
            ),
        }

    def print_summary(self):
        """Print formatted summary of metrics."""
        print("\n" + "=" * 50)
        print("DECODE METRICS SUMMARY")
        print("=" * 50)

        summary = self.summary()

        print(f"\nGeneration Stats:")
        print(f"  Total tokens generated: {summary['total_tokens_generated']}")
        print(f"  Speculation rounds: {summary['speculation_rounds']}")

        print(f"\nPerformance:")
        if summary["speedup"]:
            print(f"  Speedup: {summary['speedup']}x")
        print(f"  Acceptance rate: {summary['acceptance_rate']}%")
        print(f"  Block efficiency: {summary['block_efficiency']} tokens/round")

        print(f"\nTiming:")
        if summary["baseline_time_s"]:
            print(f"  Baseline time: {summary['baseline_time_s']}s")
        if summary["speculative_time_s"]:
            print(f"  Speculative time: {summary['speculative_time_s']}s")

        print(f"\nThroughput:")
        print(f"  Baseline: {summary['throughput_baseline_tok_per_s']} tok/s")
        print(f"  Speculative: {summary['throughput_speculative_tok_per_s']} tok/s")

        print(f"\nLatency (per token):")
        print(f"  Baseline: {summary['latency_per_token_baseline_ms']} ms")
        print(f"  Speculative: {summary['latency_per_token_speculative_ms']} ms")

        print("\n" + "=" * 50 + "\n")

    def save_summary(self, directory="results"):
        """Save metrics summary to JSON file."""
        os.makedirs(directory, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(
            directory, f"metrics_{self.summary_file_name}_{timestamp}.json"
        )

        with open(filepath, "w") as f:
            json.dump(self.summary(), f, indent=2)

        print(f"Metrics saved to {filepath}")
