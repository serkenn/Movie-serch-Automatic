"""現在の外向きIPと所在地を取得するユーティリティ。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import time

_TRAFFIC_STATE = {
    "ts": None,
    "sent": None,
    "recv": None,
}


@dataclass
class NetworkStatus:
    """ネットワーク状態のスナップショット。"""
    checked_at: str
    effective_ip: str | None
    origin_ip: str | None
    city: str | None
    region: str | None
    country: str | None
    org: str | None
    proxy_used: bool
    is_origin_ip: bool
    warning: str | None = None
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "checked_at": self.checked_at,
            "effective_ip": self.effective_ip,
            "origin_ip": self.origin_ip,
            "city": self.city,
            "region": self.region,
            "country": self.country,
            "org": self.org,
            "proxy_used": self.proxy_used,
            "is_origin_ip": self.is_origin_ip,
            "warning": self.warning,
            "error": self.error,
        }


@dataclass
class TrafficStatus:
    """通信量のスナップショット。"""
    checked_at: str
    bytes_sent_total: int
    bytes_recv_total: int
    upload_bps: float
    download_bps: float
    upload_mbps: float
    download_mbps: float
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "checked_at": self.checked_at,
            "bytes_sent_total": self.bytes_sent_total,
            "bytes_recv_total": self.bytes_recv_total,
            "upload_bps": self.upload_bps,
            "download_bps": self.download_bps,
            "upload_mbps": self.upload_mbps,
            "download_mbps": self.download_mbps,
            "error": self.error,
        }


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _build_requests_proxies(proxy: str | None) -> dict | None:
    if not proxy:
        return None
    return {"http": proxy, "https": proxy}


def _fetch_ipinfo(proxy: str | None, timeout: int = 8) -> dict:
    import requests

    proxies = _build_requests_proxies(proxy)
    errors: list[str] = []

    providers = [
        ("https://ipinfo.io/json", "ipinfo"),
        ("https://ipapi.co/json/", "ipapi"),
        ("https://ipwho.is/", "ipwhois"),
    ]

    for url, provider in providers:
        try:
            resp = requests.get(url, timeout=timeout, proxies=proxies)
            resp.raise_for_status()
            data = resp.json()

            if provider == "ipinfo":
                return data
            if provider == "ipapi":
                return {
                    "ip": data.get("ip"),
                    "city": data.get("city"),
                    "region": data.get("region"),
                    "country": data.get("country_name") or data.get("country"),
                    "org": data.get("org"),
                }
            if provider == "ipwhois":
                return {
                    "ip": data.get("ip"),
                    "city": data.get("city"),
                    "region": data.get("region"),
                    "country": data.get("country"),
                    "org": data.get("connection", {}).get("org"),
                }
        except Exception as e:
            errors.append(f"{provider}: {e}")

    raise RuntimeError(" / ".join(errors))


def get_network_status(proxy: str | None = None, expect_proxy: bool = False) -> NetworkStatus:
    """現在のIPと所在地を取得し、Origin IP警告を判定する。"""
    checked_at = _now_iso()

    try:
        origin = _fetch_ipinfo(proxy=None)
    except Exception as e:
        return NetworkStatus(
            checked_at=checked_at,
            effective_ip=None,
            origin_ip=None,
            city=None,
            region=None,
            country=None,
            org=None,
            proxy_used=bool(proxy),
            is_origin_ip=False,
            error=f"origin取得失敗: {e}",
        )

    try:
        effective = _fetch_ipinfo(proxy=proxy)
    except Exception as e:
        return NetworkStatus(
            checked_at=checked_at,
            effective_ip=None,
            origin_ip=origin.get("ip"),
            city=None,
            region=None,
            country=None,
            org=None,
            proxy_used=bool(proxy),
            is_origin_ip=False,
            error=f"effective取得失敗: {e}",
        )

    origin_ip = origin.get("ip")
    effective_ip = effective.get("ip")
    is_origin = bool(origin_ip and effective_ip and origin_ip == effective_ip)

    warning = None
    if expect_proxy and is_origin:
        warning = "Origin IP のままです。プロキシ/VPN経由になっていない可能性があります。"

    return NetworkStatus(
        checked_at=checked_at,
        effective_ip=effective_ip,
        origin_ip=origin_ip,
        city=effective.get("city"),
        region=effective.get("region"),
        country=effective.get("country"),
        org=effective.get("org"),
        proxy_used=bool(proxy),
        is_origin_ip=is_origin,
        warning=warning,
        error=None,
    )


def get_traffic_status() -> TrafficStatus:
    """上り/下り通信量（総量と瞬間レート）を返す。"""
    checked_at = _now_iso()
    try:
        import psutil
        counters = psutil.net_io_counters(pernic=False)
        now = time.monotonic()
        sent = int(counters.bytes_sent)
        recv = int(counters.bytes_recv)
    except Exception as e:
        return TrafficStatus(
            checked_at=checked_at,
            bytes_sent_total=0,
            bytes_recv_total=0,
            upload_bps=0.0,
            download_bps=0.0,
            upload_mbps=0.0,
            download_mbps=0.0,
            error=str(e),
        )

    prev_ts = _TRAFFIC_STATE["ts"]
    prev_sent = _TRAFFIC_STATE["sent"]
    prev_recv = _TRAFFIC_STATE["recv"]

    upload_bps = 0.0
    download_bps = 0.0
    if prev_ts is not None and prev_sent is not None and prev_recv is not None:
        dt = max(now - prev_ts, 1e-6)
        upload_bps = max((sent - prev_sent) / dt, 0.0)
        download_bps = max((recv - prev_recv) / dt, 0.0)

    _TRAFFIC_STATE["ts"] = now
    _TRAFFIC_STATE["sent"] = sent
    _TRAFFIC_STATE["recv"] = recv

    return TrafficStatus(
        checked_at=checked_at,
        bytes_sent_total=sent,
        bytes_recv_total=recv,
        upload_bps=upload_bps,
        download_bps=download_bps,
        upload_mbps=(upload_bps * 8.0) / 1_000_000.0,
        download_mbps=(download_bps * 8.0) / 1_000_000.0,
        error=None,
    )
