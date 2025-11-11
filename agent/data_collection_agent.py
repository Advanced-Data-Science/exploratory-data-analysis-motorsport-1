# agent/data_collection_agent.py
import os
import json
import time
import random
import logging
from datetime import datetime, timezone, timedelta, date, time as dt_time
from pathlib import Path
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

import pandas as pd
import numpy as np
import fastf1
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class DataCollectionAgent:
    # Minimal built-in mapping; prefer circuits.json aliases
    _EVENT_TO_CIRCUIT = {
        "Bahrain Grand Prix": "sakhir",
        "Saudi Arabian Grand Prix": "jeddah",
        "Australian Grand Prix": "albert_park",
        "Emilia Romagna Grand Prix": "imola",
        "Miami Grand Prix": "miami",
        "Spanish Grand Prix": "catalunya",
        "Monaco Grand Prix": "monaco",
        "Azerbaijan Grand Prix": "baku",
        "Canadian Grand Prix": "gilles_villeneuve",
        "British Grand Prix": "silverstone",
        "Austrian Grand Prix": "red_bull_ring",
        "Hungarian Grand Prix": "hungaroring",
        "Belgian Grand Prix": "spa",
        "Dutch Grand Prix": "zandvoort",
        "Italian Grand Prix": "monza",
        "Singapore Grand Prix": "marina_bay",
        "Japanese Grand Prix": "suzuka",
        "United States Grand Prix": "cota",
        "Mexico City Grand Prix": "hermanos_rodriguez",
        "São Paulo Grand Prix": "interlagos",
        "Abu Dhabi Grand Prix": "yas_marina",
        "French Grand Prix": "paul_ricard",
        "Qatar Grand Prix": "lusail",
        "Las Vegas Grand Prix": "las_vegas",
        "Chinese Grand Prix": "shanghai",
    }

    def __init__(self, config_file: str):
        self.config = self._load_json(config_file)
        self._setup_logging()

        # Paths
        paths = self.config.get("paths", {})
        self.raw_dir = Path(paths.get("raw_dir", "data/raw"))
        self.processed_dir = Path(paths.get("processed_dir", "data/processed"))
        self.metadata_dir = Path(paths.get("metadata_dir", "data/metadata"))
        self.reports_dir = Path(paths.get("reports_dir", "reports"))
        self.logs_dir = Path(paths.get("logs_dir", "logs"))
        for p in [self.raw_dir, self.processed_dir, self.metadata_dir, self.reports_dir, self.logs_dir]:
            p.mkdir(parents=True, exist_ok=True)

        # Open-Meteo config
        om_cfg = self.config.get("openmeteo", {})
        self.om_base = om_cfg.get("base_url", "https://archive-api.open-meteo.com/v1/archive")
        self.om_timezone = om_cfg.get("timezone", "UTC")
        self.om_hourly_vars = om_cfg.get(
            "hourly",
            ["temperature_2m", "relative_humidity_2m", "precipitation", "wind_speed_10m"]
        )
        self.om_timeout_seconds = int(om_cfg.get("timeout_seconds", 20))
        self.om_rate_limit_per_min = int(om_cfg.get("rate_limit_per_min", 50))

        # Run settings
        run = self.config.get("run", {})
        self.base_delay = float(self.config.get("base_delay", 0.8))
        self.drop_null_policy = set(run.get("drop_null_policy", ["temp_c", "humidity_pct", "wind_speed_ms"]))
        pad = run.get("session_window_padding_hours", {"before": 2, "after": 1})
        self.pad_before_h = int(pad.get("before", 2))
        self.pad_after_h = int(pad.get("after", 1))
        # Output filename (your requested default)
        self.weather_output_filename = run.get("weather_output_filename", "weather_conditions_openmeteo.csv")

        # Rate limit window
        self._window_start = time.time()
        self._requests_in_window = 0

        # Circuits & FastF1
        circuits_file = run.get("circuits_file", str(self.metadata_dir / "circuits.json"))
        self.circuits = self._load_json(circuits_file, default=[])
        self.logger.info(f"Loaded {len(self.circuits)} circuits from {circuits_file}")

        ff1_cfg = self.config.get("fastf1", {})
        self.ff1_enabled = bool(ff1_cfg.get("enabled", True))
        self.ff1_cache = Path(ff1_cfg.get("cache_path", "fastf1_cache"))
        self.ff1_cache.mkdir(parents=True, exist_ok=True)
        fastf1.Cache.enable_cache(str(self.ff1_cache))
        self.ff1_years = list(ff1_cfg.get("years", [2022, 2023, 2024]))
        self.ff1_events = list(ff1_cfg.get("events", []))
        self.ff1_session_types = list(ff1_cfg.get("session_types", ["FP1", "FP2", "FP3"]))
        self.ff1_filters = ff1_cfg.get("filters", {})

        # Collection state
        self.collection_stats = {
            "start_time": datetime.now(timezone.utc).isoformat(),
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "apis_used": set(["openmeteo.historical"]),
        }
        self.data_store: List[Dict[str, Any]] = []
        self.delay_multiplier = 1.0

        # Plan weather requests (session-aligned)
        self.sample_strategy = run.get("sample_strategy", "session_times")
        self._weather_plan = self._plan_build_session_times() if self.sample_strategy == "session_times" else []
        self._weather_idx = 0
        self._weather_done = False
        self._ff1_done = not self.ff1_enabled
        self.logger.info(f"Planned weather windows: {len(self._weather_plan)}")

    # ---------- Top-level run ----------
    def run_collection(self) -> None:
        self.logger.info("Starting collection (Open-Meteo only).")
        try:
            self.collect_data()
        except Exception as e:
            self.logger.exception(f"Run failed: {e}")
        finally:
            self._write_run_reports()

    def collect_data(self) -> None:
        max_req = int(self.config.get("max_requests", 0))
        target_rows = int(self.config.get("target_rows", 100000))

        while not self._complete(target_rows):
            if max_req and self.collection_stats["total_requests"] >= max_req:
                self.logger.info("Hit max_requests; stopping.")
                break

            self._check_rate_limits()

            # 1) Weather (Open-Meteo)
            self._fetch_openmeteo_window()

            # 2) Laps via FastF1 (if enabled)
            if self.ff1_enabled:
                rows = self._collect_fastf1_step()
                if rows:
                    self._store_laps(rows)

            self._sleep_with_jitter()

    # ---------- Open-Meteo ----------
    def _fetch_openmeteo_window(self) -> None:
        item = self._next_weather_plan()
        if item is None:
            self._weather_done = True
            return

        circuit = item["circuit"]
        ts_anchor = item["timestamp_utc"]  # session start UTC
        start_utc = (ts_anchor - timedelta(hours=self.pad_before_h)).replace(tzinfo=timezone.utc)
        end_utc   = (ts_anchor + timedelta(hours=self.pad_after_h)).replace(tzinfo=timezone.utc)
        start_date = start_utc.date().isoformat()
        end_date   = end_utc.date().isoformat()

        params = {
            "latitude": circuit["lat"],
            "longitude": circuit["lon"],
            "start_date": start_date,
            "end_date": end_date,
            "hourly": ",".join(self.om_hourly_vars),
            "timezone": self.om_timezone,
        }

        self.collection_stats["total_requests"] += 1
        self._requests_in_window += 1

        resp = self._safe_get(self.om_base, params, timeout=self.om_timeout_seconds)
        if resp is None or resp.status_code != 200:
            self.collection_stats["failed_requests"] += 1
            self.logger.warning(f"Open-Meteo error for {circuit['circuit_id']} @ {ts_anchor}: "
                               f"{'no response' if resp is None else resp.status_code}")
            return

        try:
            payload = resp.json()
        except Exception as e:
            self.collection_stats["failed_requests"] += 1
            self.logger.warning(f"Open-Meteo JSON parse error: {e}")
            return

        self.collection_stats["successful_requests"] += 1
        self._write_raw_json(circuit["circuit_id"], ts_anchor, payload)

        hourly = payload.get("hourly", {})
        times = hourly.get("time", [])
        if not times:
            self.logger.info(f"No hourly data for {circuit['circuit_id']} {start_date}->{end_date}")
            return

        df = pd.DataFrame({"timestamp_utc": pd.to_datetime(times, utc=True)})

        def arr(key):
            a = hourly.get(key)
            return a if isinstance(a, list) and len(a) == len(times) else [None] * len(times)

        df["temp_c"] = arr("temperature_2m")
        df["humidity_pct"] = arr("relative_humidity_2m")
        df["precip_mm"] = arr("precipitation")
        wind_kmh = arr("wind_speed_10m")
        df["wind_speed_ms"] = [None if v is None else float(v) / 3.6 for v in wind_kmh]

        mask = (df["timestamp_utc"] >= pd.Timestamp(start_utc)) & (df["timestamp_utc"] <= pd.Timestamp(end_utc))
        df = df.loc[mask].copy()
        if df.empty:
            self.logger.info(f"No rows within session window for {circuit['circuit_id']} [{start_utc}..{end_utc}]")
            return

        tz = ZoneInfo(self._circuit_tz(circuit["circuit_id"]))
        for _, r in df.iterrows():
            ts = r["timestamp_utc"].to_pydatetime()
            local_dt = ts.astimezone(tz)
            out = {
                "circuit_id": circuit["circuit_id"],
                "timestamp_utc": ts.isoformat(),
                "local_time": local_dt.isoformat(),
                "local_hour": int(local_dt.hour),
                "temp_c": None if pd.isna(r["temp_c"]) else float(r["temp_c"]),
                "humidity_pct": None if pd.isna(r["humidity_pct"]) else float(r["humidity_pct"]),
                "precip_mm": None if pd.isna(r["precip_mm"]) else float(r["precip_mm"]),
                "wind_speed_ms": None if pd.isna(r["wind_speed_ms"]) else float(r["wind_speed_ms"]),
                "weather_main": "",
                "weather_desc": "",
                "source": "openmeteo.historical",
                "ingested_at": datetime.now(timezone.utc).isoformat(),
            }
            if self._valid_row(out):
                self._store_weather(out)

    # ---------- FastF1 ----------
    def _collect_fastf1_step(self) -> List[Dict[str, Any]]:
        plan = self._ff1_next_plan()
        if plan is None:
            self._ff1_done = True
            return []

        y, ev, st = plan["year"], plan["event"], plan["session"]
        self.logger.info(f"FastF1: loading {y} | {ev} | {st}")

        try:
            ses = fastf1.get_session(y, ev, st)
            ses.load(laps=True, telemetry=False, weather=True)
        except Exception as e:
            self.logger.warning(f"FastF1 load failed {y} {ev} {st}: {e}")
            return []

        rows = self._laps_to_rows(ses)
        # Persist raw laps for audit
        try:
            raw_folder = self.raw_dir / "fastf1"
            raw_folder.mkdir(parents=True, exist_ok=True)
            (raw_folder / f"laps_{y}_{ev.replace(' ','_')}_{st}.csv").write_text(
                ses.laps.to_csv(index=False), encoding="utf-8"
            )
        except Exception:
            pass
        return rows

    def _laps_to_rows(self, session):
        laps = session.laps
        if laps is None or laps.empty:
            return []

        df = laps.copy()
        if "LapTime" in df.columns:
            df["lap_time_s"] = pd.to_timedelta(df["LapTime"]).dt.total_seconds()
        else:
            df["lap_time_s"] = np.nan

        f = self.ff1_filters or {}
        if f.get("exclude_outlaps", True) and "PitOutTime" in df.columns:
            df = df[df["PitOutTime"].isna()]
        if f.get("exclude_inlaps", True) and "PitInTime" in df.columns:
            df = df[df["PitInTime"].isna()]
        if f.get("exclude_yellow_vsc_sc_red", False) and "TrackStatus" in df.columns:
            df = df[df["TrackStatus"].isin(["1", 1, np.nan, None, ""])]

        if df["lap_time_s"].notna().any():
            best = df["lap_time_s"].min()
            limit = best + float(f.get("delta_to_session_best_sec", 8.0))
            df = df[(df["lap_time_s"].notna()) & (df["lap_time_s"] <= limit)]

        cols_map = {
            "LapNumber": "lap_number",
            "Driver": "driver",
            "Team": "team",
            "Stint": "stint_number",
            "Compound": "compound_code",
            "TyreLife": "tyre_age_laps",
            "IsPersonalBest": "is_pb",
            "TrackStatus": "track_status",
        }
        keep = [k for k in cols_map.keys() if k in df.columns] + ["lap_time_s"]
        out = df[keep].rename(columns=cols_map).copy()

        out["year"] = int(session.event.year)
        out["gp_name"] = str(session.event["EventName"])
        out["session"] = str(session.name)
        out["timestamp_utc"] = self._session_utc_iso(session)
        out["source"] = "fastf1.laps"

        # numeric casts
        for k in ["lap_time_s", "tyre_age_laps", "lap_number"]:
            if k in out.columns:
                out[k] = pd.to_numeric(out[k], errors="coerce")

        return out.to_dict(orient="records")

    # ---------- Storage ----------
    def _store_weather(self, row: Dict[str, Any]) -> None:
        self.data_store.append(row)
        out_file = self.weather_output_filename or "weather_conditions_openmeteo.csv"
        out_path = (self.processed_dir / out_file).resolve()

        header = [
            "circuit_id","timestamp_utc","local_time","local_hour","temp_c","humidity_pct",
            "precip_mm","wind_speed_ms","weather_main","weather_desc","source","ingested_at"
        ]

        # fast dedupe on (circuit_id, timestamp_utc)
        exists = set()
        if out_path.exists():
            try:
                df = pd.read_csv(out_path, usecols=["circuit_id","timestamp_utc"])
                exists = set(zip(df["circuit_id"].astype(str), df["timestamp_utc"].astype(str)))
            except Exception:
                pass

        key = (str(row["circuit_id"]), str(row["timestamp_utc"]))
        if key in exists:
            return

        line = ",".join(str(row.get(c, "")) for c in header)
        if not out_path.exists():
            out_path.write_text(",".join(header) + "\n" + line + "\n", encoding="utf-8")
        else:
            with out_path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")

        if self.config.get("run", {}).get("dedupe_on_write", True):
            try:
                df = pd.read_csv(out_path)
                df = df.drop_duplicates(subset=["circuit_id", "timestamp_utc"])
                df.to_csv(out_path, index=False)
            except Exception:
                pass

    def _store_laps(self, rows: List[Dict[str, Any]]) -> None:
        if not rows:
            return

        out_path = (self.processed_dir / "laps.csv").resolve()
        cols = [
            "year","gp_name","session","driver","team","stint_number","compound_code",
            "tyre_age_laps","lap_number","lap_time_s","track_status","is_pb",
            "timestamp_utc","source"
        ]

        existing_keys = set()
        if out_path.exists():
            try:
                use = ["year","gp_name","session","driver","lap_number"]
                df = pd.read_csv(out_path, usecols=use, dtype={"year": int, "lap_number": float})
                df["lap_number"] = df["lap_number"].astype(pd.Int64Dtype()).astype(str)
                existing_keys = set(zip(
                    df["year"].astype(str),
                    df["gp_name"].astype(str),
                    df["session"].astype(str),
                    df["driver"].astype(str),
                    df["lap_number"].astype(str),
                ))
            except Exception:
                pass

        def k(r):
            return (
                str(r.get("year","")), str(r.get("gp_name","")), str(r.get("session","")),
                str(r.get("driver","")), str(r.get("lap_number",""))
            )

        new_rows = [r for r in rows if k(r) not in existing_keys]
        if not new_rows:
            return

        header_needed = not out_path.exists()
        if header_needed:
            with out_path.open("w", encoding="utf-8") as f:
                f.write(",".join(cols) + "\n")

        with out_path.open("a", encoding="utf-8") as f:
            for r in new_rows:
                f.write(",".join("" if pd.isna(r.get(c, None)) else str(r.get(c, "")) for c in cols) + "\n")

        if self.config.get("run", {}).get("dedupe_on_write", True):
            try:
                df = pd.read_csv(out_path)
                df = df.drop_duplicates(subset=["year","gp_name","session","driver","lap_number"])
                df.to_csv(out_path, index=False)
            except Exception:
                pass

    # ---------- Planning ----------
    def _plan_build_session_times(self) -> List[Dict[str, Any]]:
        plan: List[Dict[str, Any]] = []
        years = list(self.config.get("run", {}).get("seasons", self.ff1_years))
        wanted = set(s.upper() for s in (self.ff1_session_types or ["FP1", "FP2", "FP3"]))

        for y in years:
            try:
                schedule = fastf1.get_event_schedule(y, include_testing=False)
            except Exception as e:
                self.logger.warning(f"Schedule load failed for {y}: {e}")
                continue

            session_cols = [c for c in schedule.columns if str(c).lower().startswith("session")]
            for _, row in schedule.iterrows():
                ev = str(row.get("EventName") or "").strip()
                if not ev:
                    continue

                cid = self._event_to_circuit_id_fallback(ev)
                circuit = next((c for c in self.circuits if c.get("circuit_id") == cid), None)
                if not circuit:
                    self.logger.warning(f"No circuit object for event '{ev}' ({y}). Add to circuits.json aliases.")
                    continue

                present = set()
                for c in session_cols:
                    name = str(row.get(c) or "").lower()
                    if "practice 1" in name: present.add("FP1")
                    if "practice 2" in name: present.add("FP2")
                    if "practice 3" in name: present.add("FP3")

                for code in sorted(wanted & present):
                    try:
                        ses = fastf1.get_session(y, ev, code)
                        ts_iso = self._session_utc_iso(ses)
                        ts = pd.to_datetime(ts_iso, utc=True).to_pydatetime()
                    except Exception as e:
                        self.logger.warning(f"Resolve session time failed {y} {ev} {code}: {e}")
                        continue

                    plan.append({
                        "year": int(y),
                        "event": ev,
                        "session": code,
                        "circuit_id": cid,
                        "circuit": circuit,
                        "timestamp_utc": ts,
                    })

        # Deduplicate & resume if already collected
        plan.sort(key=lambda x: (x["year"], x["event"], x["session"]))
        uniq = {(p["year"], p["event"], p["session"], p["circuit_id"], p["timestamp_utc"]): p for p in plan}
        plan = list(uniq.values())

        existing = self._existing_weather_keys()
        filtered = [p for p in plan if (p["circuit_id"], p["timestamp_utc"].isoformat()) not in existing]
        self.logger.info(f"Weather plan built: {len(filtered)} items after resume filter (from {len(plan)})")
        return filtered

    def _next_weather_plan(self) -> Optional[Dict[str, Any]]:
        if self._weather_idx < len(self._weather_plan):
            item = self._weather_plan[self._weather_idx]
            self._weather_idx += 1
            return item
        return None

    def _ff1_next_plan(self):
        if not hasattr(self, "_ff1_plan"):
            self._ff1_plan = self._ff1_plan_sessions()
            self._ff1_idx = 0
        if self._ff1_idx < len(self._ff1_plan):
            i = self._ff1_plan[self._ff1_idx]
            self._ff1_idx += 1
            return i
        return None

    def _ff1_plan_sessions(self):
        if not self.ff1_enabled:
            return []
        years = list(self.ff1_years or [])
        events_filter = set(self.ff1_events or [])
        wanted = set(s.upper() for s in (self.ff1_session_types or ["FP1","FP2","FP3"]))
        plan = []

        for y in years:
            try:
                sched = fastf1.get_event_schedule(y, include_testing=False)
            except Exception:
                continue
            df = sched[sched["EventName"].isin(events_filter)] if events_filter else sched
            sess_cols = [c for c in df.columns if str(c).lower().startswith("session")]
            for _, row in df.iterrows():
                ev = str(row.get("EventName") or "")
                present = set()
                for c in sess_cols:
                    n = str(row.get(c) or "").lower()
                    if "practice 1" in n: present.add("FP1")
                    if "practice 2" in n: present.add("FP2")
                    if "practice 3" in n: present.add("FP3")
                for s in sorted(wanted & present):
                    plan.append({"year": int(y), "event": ev, "session": s})

        # resume filter
        existing = self._existing_session_keys()
        plan = [p for p in plan if (p["year"], p["event"], p["session"]) not in existing]
        plan.sort(key=lambda x: (x["year"], x["event"], x["session"]))
        return plan

    # ---------- Reporting ----------
    def _write_run_reports(self):
        self.collection_stats["end_time"] = datetime.now(timezone.utc).isoformat()
        self.collection_stats["apis_used"] = sorted(self.collection_stats["apis_used"])
        self.collection_stats["rows"] = len(self.data_store)

        # JSON report
        report = {
            "summary": self.collection_stats,
            "coverage": self._coverage_summary(),
            "joinability": self._joinability_summary(),
            "metrics": {
                "completeness": self._completeness_rate(),
                "consistency": self._consistency_rate(),
            },
            "sources": {
                "openmeteo": {
                    "base_url": self.om_base,
                    "timezone": self.om_timezone,
                    "hourly_vars": self.om_hourly_vars,
                }
            },
        }
        (self.reports_dir / "quality_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

        # Minimal HTML
        html = f"""<!doctype html><html><head><meta charset="utf-8"><title>Quality Report</title></head>
<body>
<h1>Dataset Quality Report</h1>
<pre>{json.dumps(report, indent=2)}</pre>
</body></html>"""
        (self.reports_dir / "quality_report.html").write_text(html, encoding="utf-8")

        # Append run history
        with (self.metadata_dir / "run_history.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(self.collection_stats) + "\n")

    # ---------- Helpers ----------
    def _setup_logging(self):
        self.logs_dir = Path(self.config.get("paths", {}).get("logs_dir", "logs"))
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(self.logs_dir / "collection.log"), logging.StreamHandler()],
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("Logger ready.")

    def _load_json(self, path: str, default=None):
        p = Path(path)
        if not p.exists():
            return default if default is not None else {}
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _safe_get(self, url: str, params: Dict[str, Any], timeout: int) -> Optional[requests.Response]:
        try:
            return requests.get(url, params=params, timeout=timeout)
        except Exception as e:
            self.logger.warning(f"HTTP error: {e}")
            return None

    def _write_raw_json(self, circuit_id: str, ts: datetime, payload: Dict[str, Any]) -> None:
        run_folder = self.raw_dir / datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
        run_folder.mkdir(parents=True, exist_ok=True)
        (run_folder / f"wx_openmeteo_{circuit_id}_{ts.strftime('%Y%m%d%H')}.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )

    def _circuit_tz(self, circuit_id: str) -> str:
        for c in self.circuits:
            if c.get("circuit_id") == circuit_id:
                return c.get("tz", c.get("timezone", "UTC"))
        return "UTC"

    def _event_to_circuit_id_fallback(self, event_name: str) -> Optional[str]:
        ev = (event_name or "").strip().lower()
        # Try circuits.json aliases
        for c in self.circuits:
            aliases = [a.strip().lower() for a in (c.get("aliases") or [])]
            if ev in aliases or any(ev in a or a in ev for a in aliases):
                return c.get("circuit_id")
        return self._EVENT_TO_CIRCUIT.get(event_name)

    def _session_utc_iso(self, session) -> str:
        try:
            dt = pd.to_datetime(session.date)
            tz_name = (session.event.get('Timezone') or session.event.get('timezone') or 'UTC')
            if getattr(dt, 'tzinfo', None) is not None:
                ts = dt.tz_convert('UTC')
            else:
                ts = dt.tz_localize(tz_name).tz_convert('UTC')
            return ts.isoformat()
        except Exception:
            return pd.Timestamp.utcnow().tz_localize('UTC').isoformat()

    def _completeness_rate(self) -> float:
        if not self.data_store:
            return 0.0
        req = self.drop_null_policy
        ok = 0
        for r in self.data_store:
            if all(r.get(k) is not None for k in req):
                ok += 1
        return ok / len(self.data_store)

    def _consistency_rate(self) -> float:
        if not self.data_store:
            return 0.0
        seen = set()
        unique = 0
        for r in self.data_store:
            key = (r.get("circuit_id"), r.get("timestamp_utc"))
            if key not in seen:
                seen.add(key)
                unique += 1
        return unique / len(self.data_store)

    def _coverage_summary(self) -> Dict[str, Any]:
        wx, lp = self._load_processed_frames()
        years = self.config.get("run", {}).get("seasons", self.ff1_years)
        sessions_expected = 0
        sessions_with_any = 0

        # Build expected sessions via schedule count
        try:
            for y in years:
                sch = fastf1.get_event_schedule(y, include_testing=False)
                sess_cols = [c for c in sch.columns if str(c).lower().startswith("session")]
                for _, row in sch.iterrows():
                    present = []
                    for c in sess_cols:
                        n = str(row.get(c) or "").lower()
                        if "practice 1" in n: present.append("FP1")
                        if "practice 2" in n: present.append("FP2")
                        if "practice 3" in n: present.append("FP3")
                    sessions_expected += sum(s in self.ff1_session_types for s in present)
        except Exception:
            pass

        if not wx.empty:
            sessions_with_any = int(
                wx.groupby(["circuit_id", pd.to_datetime(wx["timestamp_utc"]).dt.date]).ngroups
            )

        return {"sessions_expected_estimate": sessions_expected, "sessions_with_weather_any": sessions_with_any}

    def _joinability_summary(self, tolerance_minutes: int = 60) -> Dict[str, Any]:
        wx, lp = self._load_processed_frames()
        if wx.empty or lp.empty:
            return {"joinable_rate": 0.0, "checked_sessions": 0}

        sess_df = (
            lp[["year", "gp_name", "session", "timestamp_utc"]]
            .drop_duplicates()
            .rename(columns={"timestamp_utc": "ses_ts"})
        )
        sess_df["ses_ts"] = pd.to_datetime(sess_df["ses_ts"], utc=True)
        e2c = self._EVENT_TO_CIRCUIT.copy()
        sess_df["circuit_id"] = sess_df["gp_name"].map(e2c)

        wx["ts_utc"] = pd.to_datetime(wx["timestamp_utc"], utc=True)
        tol = pd.Timedelta(minutes=tolerance_minutes)

        joinable = 0
        for _, r in sess_df.iterrows():
            cid = r["circuit_id"]
            if not cid:
                continue
            nearby = wx[(wx["circuit_id"] == cid) & (wx["ts_utc"].between(r["ses_ts"] - tol, r["ses_ts"] + tol))]
            if not nearby.empty:
                joinable += 1

        rate = joinable / max(len(sess_df), 1)
        return {"joinable_rate": rate, "checked_sessions": int(len(sess_df))}

    def _load_processed_frames(self):
        wx_path = self.processed_dir / (self.weather_output_filename or "weather_conditions_openmeteo.csv")
        lp_path = self.processed_dir / "laps.csv"
        wx = pd.read_csv(wx_path) if wx_path.exists() else pd.DataFrame()
        lp = pd.read_csv(lp_path) if lp_path.exists() else pd.DataFrame()
        return wx, lp

    def _valid_row(self, r: Dict[str, Any]) -> bool:
        try:
            t = r.get("temp_c"); h = r.get("humidity_pct"); w = r.get("wind_speed_ms")
            if t is not None and not (-60 <= float(t) <= 60): return False
            if h is not None and not (0 <= float(h) <= 100):  return False
            if w is not None and not (0 <= float(w) <= 100):  return False
            return True
        except Exception:
            return False

    def _check_rate_limits(self):
        now = time.time()
        elapsed = now - self._window_start
        if elapsed >= 60:
            self._window_start = now
            self._requests_in_window = 0
        if self._requests_in_window >= self.om_rate_limit_per_min:
            sleep = max(0.0, 60 - elapsed) + 0.1
            self.logger.info(f"Rate limit reached; sleeping {sleep:.1f}s.")
            time.sleep(sleep)
            self._window_start = time.time()
            self._requests_in_window = 0

    def _sleep_with_jitter(self):
        delay = self.base_delay * self.delay_multiplier
        time.sleep(max(0.0, delay * random.uniform(0.5, 1.5)))

    def _complete(self, target_rows: int) -> bool:
        done_weather = self._weather_done or (self._weather_idx >= len(self._weather_plan))
        return (done_weather and (not self.ff1_enabled or self._ff1_done)) or (len(self.data_store) >= target_rows)

    def _existing_weather_keys(self) -> set[tuple[str, str]]:
        p = self.processed_dir / (self.weather_output_filename or "weather_conditions_openmeteo.csv")
        if not p.exists():
            return set()
        try:
            df = pd.read_csv(p, usecols=["circuit_id", "timestamp_utc"]).drop_duplicates()
            return set(zip(df["circuit_id"].astype(str), df["timestamp_utc"].astype(str)))
        except Exception:
            return set()

    def _existing_session_keys(self) -> set[tuple[int, str, str]]:
        p = self.processed_dir / "laps.csv"
        if not p.exists():
            return set()
        try:
            df = pd.read_csv(p, usecols=["year", "gp_name", "session"]).drop_duplicates()
            df["year"] = df["year"].astype(int)
            df["gp_name"] = df["gp_name"].astype(str)
            df["session"] = df["session"].astype(str)
            return set(zip(df["year"], df["gp_name"], df["session"]))
        except Exception:
            return set()


if __name__ == "__main__":
    default_cfg = Path(__file__).with_name("config.json")
    agent = DataCollectionAgent(str(default_cfg))
    agent.run_collection()
