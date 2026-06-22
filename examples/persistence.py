"""Persistence — auto-save tracking data and request logs to poolgate_data/.

Set POOLGATE_DATA_DIR and PoolGate writes all tracking state and request
details to structured JSON files automatically — no manual PersistenceService
setup required.

Layout created under POOLGATE_DATA_DIR:
    tracking/
        usage.json       daily request counts (requests, success, failures)
        tokens.json      per-model token usage by day
        account.json     per-key token consumption by day
    requests/
        YYYY-MM-DD.jsonl one JSON line per request with full execution details
    logs/
        general.log      all log messages
        info.log         INFO and above
        error.log        ERROR and above
        request.log      request lifecycle events only

Call flush_tracking() to save in-memory state to disk.  Data is reloaded
automatically the next time GroqService() is constructed with the same
POOLGATE_DATA_DIR.

Environment (.env or shell):
    TOTAL_GROQ_KEYS=1
    GROQ_API_KEY_01=gsk_...
    POOLGATE_DATA_DIR=./poolgate_data
"""

from __future__ import annotations

import glob
import json
import os

from dotenv import load_dotenv

from exceptions import GroqServiceError
from services.provider_service import GroqService


load_dotenv()
os.environ.setdefault("POOLGATE_DATA_DIR", "./poolgate_data")


def _show_tracking_files(data_dir: str) -> None:
	"""Print a summary of each tracking JSON file that was written."""
	tracking_dir = os.path.join(data_dir, "tracking")
	for path in sorted(glob.glob(os.path.join(tracking_dir, "*.json"))):
		name = os.path.basename(path)
		with open(path) as f:
			data = json.load(f)
		days = sorted(data.keys())
		print(f"  {name}: {len(days)} day(s) — {days}")


def _show_request_journal(data_dir: str) -> None:
	"""Print the most recent entries from today's request journal."""
	requests_dir = os.path.join(data_dir, "requests")
	journals = sorted(glob.glob(os.path.join(requests_dir, "*.jsonl")))
	if not journals:
		print("  (no request journal entries yet)")
		return
	latest = journals[-1]
	print(f"  {os.path.basename(latest)}:")
	with open(latest) as f:
		lines = f.readlines()
	for line in lines:
		line = line.strip()
		if not line:  # ← skip blank lines (EOF newline, gaps, etc.)
			continue
		try:
			entry = json.loads(line)
		except json.JSONDecodeError as e:
			print(f"    [skipped malformed entry: {e}]")
			continue
		print(
			f"    {entry.get('timestamp')}  {entry.get('model')}  "
			f"tokens={entry.get('total_tokens')}  "
			f"latency={entry.get('latency_seconds')}s  "
			f"success={entry.get('success')}",
			)


def main() -> None:
	# --- First run: make some requests and flush ---
	service = GroqService()
	data_dir = service._config.data_dir

	print(f"Data directory: {data_dir}")
	print()

	response = service.invoke(
		"Say 'hello' in three languages.",
		model="llama-3.3-70b-versatile",
		)
	print(f"Response: {response.text[:80]}...")
	print(
		f"Tokens:   {response.usage.prompt_tokens} in, "
		f"{response.usage.completion_tokens} out",
		)
	print()

	# Flush in-memory counters to disk
	service.flush_tracking()
	print("Tracking flushed to disk.")
	print()

	# --- Show what was written ---
	if data_dir:
		print("Tracking files:")
		_show_tracking_files(data_dir)
		print()
		print("Request journal (latest entries):")
		_show_request_journal(data_dir)
		print()

	# --- Second run: reload verifies state survives across restarts ---
	print("Simulating restart (constructing a new GroqService)...")
	service2 = GroqService()
	snap = service2._tracking.usage_tracker.snapshot_for_day()
	print(
		f"Reloaded today's usage: "
		f"{snap.get('requests', 0)} requests, "
		f"{snap.get('successful_requests', 0)} successful",
		)

	if data_dir:
		print(f"\nLogs are at {service._config.log_dir}/")


if __name__ == "__main__":
	try:
		main()
	except GroqServiceError as e:
		print(f"Error: {e}")
