"""Rate limiting monitoring utilities for KokoroTTS API.

This module provides utilities for monitoring rate limiting behavior,
analyzing usage patterns, and generating reports.
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict, deque
import logging

from ..config import get_config

logger = logging.getLogger(__name__)


class RateLimitMonitor:
    """Monitor rate limiting metrics and usage patterns."""
    
    def __init__(self):
        self.config = get_config()
        self._usage_history = deque(maxlen=1000)  # Keep last 1000 requests
        self._violation_history = deque(maxlen=500)  # Keep last 500 violations
        self._ip_stats = defaultdict(lambda: {
            "total_requests": 0,
            "blocked_requests": 0,
            "total_characters": 0,
            "first_seen": None,
            "last_seen": None,
            "violation_count": 0
        })
    
    def record_request(self, ip: str, endpoint: str, character_count: int = 0, 
                      blocked: bool = False, block_reason: str = None):
        """Record a request for monitoring."""
        timestamp = time.time()
        
        # Record in usage history
        request_data = {
            "timestamp": timestamp,
            "ip": ip,
            "endpoint": endpoint,
            "character_count": character_count,
            "blocked": blocked,
            "block_reason": block_reason
        }
        self._usage_history.append(request_data)
        
        # Update IP statistics
        ip_stat = self._ip_stats[ip]
        ip_stat["total_requests"] += 1
        ip_stat["total_characters"] += character_count
        ip_stat["last_seen"] = timestamp
        
        if ip_stat["first_seen"] is None:
            ip_stat["first_seen"] = timestamp
        
        if blocked:
            ip_stat["blocked_requests"] += 1
            ip_stat["violation_count"] += 1
            
            # Record violation
            violation_data = {
                "timestamp": timestamp,
                "ip": ip,
                "endpoint": endpoint,
                "reason": block_reason,
                "character_count": character_count
            }
            self._violation_history.append(violation_data)
    
    def get_usage_stats(self, time_window_hours: int = 24) -> Dict[str, Any]:
        """Get usage statistics for a time window."""
        cutoff_time = time.time() - (time_window_hours * 3600)
        
        # Filter recent requests
        recent_requests = [r for r in self._usage_history if r["timestamp"] > cutoff_time]
        
        if not recent_requests:
            return {
                "time_window_hours": time_window_hours,
                "total_requests": 0,
                "blocked_requests": 0,
                "unique_ips": 0,
                "total_characters": 0,
                "requests_per_hour": 0,
                "block_rate": 0.0,
                "top_endpoints": [],
                "top_ips": []
            }
        
        # Calculate statistics
        total_requests = len(recent_requests)
        blocked_requests = sum(1 for r in recent_requests if r["blocked"])
        unique_ips = len(set(r["ip"] for r in recent_requests))
        total_characters = sum(r["character_count"] for r in recent_requests)
        
        # Calculate rates
        requests_per_hour = total_requests / time_window_hours if time_window_hours > 0 else 0
        block_rate = (blocked_requests / total_requests * 100) if total_requests > 0 else 0
        
        # Top endpoints
        endpoint_counts = defaultdict(int)
        for r in recent_requests:
            endpoint_counts[r["endpoint"]] += 1
        top_endpoints = sorted(endpoint_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Top IPs by request count
        ip_counts = defaultdict(int)
        for r in recent_requests:
            ip_counts[r["ip"]] += 1
        top_ips = sorted(ip_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            "time_window_hours": time_window_hours,
            "total_requests": total_requests,
            "blocked_requests": blocked_requests,
            "unique_ips": unique_ips,
            "total_characters": total_characters,
            "requests_per_hour": round(requests_per_hour, 2),
            "characters_per_hour": round(total_characters / time_window_hours, 2) if time_window_hours > 0 else 0,
            "block_rate": round(block_rate, 2),
            "top_endpoints": [{"endpoint": ep, "count": count} for ep, count in top_endpoints],
            "top_ips": [{"ip": ip, "count": count} for ip, count in top_ips]
        }
    
    def get_violation_report(self, time_window_hours: int = 24) -> Dict[str, Any]:
        """Get detailed violation report."""
        cutoff_time = time.time() - (time_window_hours * 3600)
        
        # Filter recent violations
        recent_violations = [v for v in self._violation_history if v["timestamp"] > cutoff_time]
        
        if not recent_violations:
            return {
                "time_window_hours": time_window_hours,
                "total_violations": 0,
                "unique_violating_ips": 0,
                "violations_by_reason": {},
                "violations_by_hour": [],
                "top_violating_ips": []
            }
        
        # Violations by reason
        violations_by_reason = defaultdict(int)
        for v in recent_violations:
            violations_by_reason[v["reason"]] += 1
        
        # Violations by hour
        violations_by_hour = defaultdict(int)
        for v in recent_violations:
            hour = datetime.fromtimestamp(v["timestamp"]).strftime("%Y-%m-%d %H:00")
            violations_by_hour[hour] += 1
        
        # Convert to sorted list
        violations_by_hour_list = [
            {"hour": hour, "count": count}
            for hour, count in sorted(violations_by_hour.items())
        ]
        
        # Top violating IPs
        ip_violation_counts = defaultdict(int)
        for v in recent_violations:
            ip_violation_counts[v["ip"]] += 1
        top_violating_ips = sorted(ip_violation_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            "time_window_hours": time_window_hours,
            "total_violations": len(recent_violations),
            "unique_violating_ips": len(ip_violation_counts),
            "violations_by_reason": dict(violations_by_reason),
            "violations_by_hour": violations_by_hour_list,
            "top_violating_ips": [{"ip": ip, "violations": count} for ip, count in top_violating_ips]
        }
    
    def get_ip_details(self, ip: str) -> Dict[str, Any]:
        """Get detailed information about a specific IP."""
        if ip not in self._ip_stats:
            return {"error": f"No data found for IP {ip}"}
        
        ip_stat = self._ip_stats[ip]
        
        # Get recent requests from this IP
        recent_requests = [r for r in self._usage_history if r["ip"] == ip]
        recent_violations = [v for v in self._violation_history if v["ip"] == ip]
        
        # Calculate additional metrics
        total_requests = ip_stat["total_requests"]
        blocked_requests = ip_stat["blocked_requests"]
        success_rate = ((total_requests - blocked_requests) / total_requests * 100) if total_requests > 0 else 0
        
        # Request pattern analysis
        if recent_requests:
            request_times = [r["timestamp"] for r in recent_requests]
            time_span = max(request_times) - min(request_times)
            avg_request_interval = time_span / len(request_times) if len(request_times) > 1 else 0
        else:
            avg_request_interval = 0
        
        return {
            "ip": ip,
            "total_requests": total_requests,
            "blocked_requests": blocked_requests,
            "success_rate": round(success_rate, 2),
            "total_characters": ip_stat["total_characters"],
            "violation_count": ip_stat["violation_count"],
            "first_seen": datetime.fromtimestamp(ip_stat["first_seen"]).isoformat() if ip_stat["first_seen"] else None,
            "last_seen": datetime.fromtimestamp(ip_stat["last_seen"]).isoformat() if ip_stat["last_seen"] else None,
            "avg_request_interval_seconds": round(avg_request_interval, 2),
            "recent_violations": [
                {
                    "timestamp": datetime.fromtimestamp(v["timestamp"]).isoformat(),
                    "reason": v["reason"],
                    "endpoint": v["endpoint"]
                }
                for v in recent_violations[-10:]  # Last 10 violations
            ]
        }
    
    def detect_abuse_patterns(self) -> List[Dict[str, Any]]:
        """Detect potential abuse patterns."""
        patterns = []
        current_time = time.time()
        
        # Check for IPs with high violation rates
        for ip, stats in self._ip_stats.items():
            if stats["total_requests"] >= 10:  # Only check IPs with significant activity
                violation_rate = (stats["blocked_requests"] / stats["total_requests"]) * 100
                
                if violation_rate > 50:  # More than 50% of requests blocked
                    patterns.append({
                        "type": "high_violation_rate",
                        "ip": ip,
                        "violation_rate": round(violation_rate, 2),
                        "total_requests": stats["total_requests"],
                        "severity": "high" if violation_rate > 80 else "medium"
                    })
        
        # Check for burst patterns (many requests in short time)
        recent_requests = [r for r in self._usage_history if current_time - r["timestamp"] < 300]  # Last 5 minutes
        ip_recent_counts = defaultdict(int)
        for r in recent_requests:
            ip_recent_counts[r["ip"]] += 1
        
        for ip, count in ip_recent_counts.items():
            if count > 50:  # More than 50 requests in 5 minutes
                patterns.append({
                    "type": "burst_pattern",
                    "ip": ip,
                    "requests_in_5min": count,
                    "severity": "high" if count > 100 else "medium"
                })
        
        # Check for character abuse (very long requests)
        recent_high_char_requests = [
            r for r in self._usage_history 
            if current_time - r["timestamp"] < 3600 and r["character_count"] > 4000
        ]
        
        ip_high_char_counts = defaultdict(int)
        for r in recent_high_char_requests:
            ip_high_char_counts[r["ip"]] += 1
        
        for ip, count in ip_high_char_counts.items():
            if count > 5:  # More than 5 high-character requests in an hour
                patterns.append({
                    "type": "character_abuse",
                    "ip": ip,
                    "high_char_requests_in_hour": count,
                    "severity": "medium"
                })
        
        return sorted(patterns, key=lambda x: x["severity"], reverse=True)
    
    def export_metrics(self, format_type: str = "json") -> str:
        """Export monitoring data in various formats."""
        data = {
            "export_timestamp": datetime.now().isoformat(),
            "usage_stats_24h": self.get_usage_stats(24),
            "usage_stats_1h": self.get_usage_stats(1),
            "violation_report_24h": self.get_violation_report(24),
            "abuse_patterns": self.detect_abuse_patterns(),
            "configuration": {
                "rate_limiting_enabled": self.config.rate_limit.is_enabled(),
                "deployment_mode": self.config.rate_limit.deployment_mode.value,
                "limits": {
                    "requests_per_minute": self.config.rate_limit.requests_per_minute,
                    "requests_per_hour": self.config.rate_limit.requests_per_hour,
                    "characters_per_request": self.config.rate_limit.max_characters_per_request,
                    "characters_per_hour": self.config.rate_limit.characters_per_hour,
                    "max_concurrent": self.config.rate_limit.max_concurrent_requests
                }
            }
        }
        
        if format_type.lower() == "json":
            return json.dumps(data, indent=2)
        elif format_type.lower() == "csv":
            # Basic CSV export for usage data
            lines = ["timestamp,ip,endpoint,character_count,blocked,block_reason"]
            for request in self._usage_history:
                lines.append(f"{request['timestamp']},{request['ip']},{request['endpoint']},{request['character_count']},{request['blocked']},{request.get('block_reason', '')}")
            return "\n".join(lines)
        else:
            raise ValueError(f"Unsupported format: {format_type}")


# Global monitor instance
_monitor = None

def get_monitor() -> RateLimitMonitor:
    """Get the global rate limit monitor instance."""
    global _monitor
    if _monitor is None:
        _monitor = RateLimitMonitor()
    return _monitor

async def cleanup_old_data():
    """Cleanup old monitoring data to prevent memory leaks."""
    monitor = get_monitor()
    
    # This function could be called periodically to clean up old data
    # For now, the deque maxlen handles this automatically
    logger.info("Rate limit monitoring data cleanup completed")

def record_request_async(ip: str, endpoint: str, character_count: int = 0, 
                        blocked: bool = False, block_reason: str = None):
    """Non-blocking wrapper to record request."""
    try:
        monitor = get_monitor()
        monitor.record_request(ip, endpoint, character_count, blocked, block_reason)
    except Exception as e:
        logger.error(f"Failed to record request for monitoring: {e}")

# Convenience functions for common monitoring tasks
def get_current_usage_stats() -> Dict[str, Any]:
    """Get current usage statistics."""
    return get_monitor().get_usage_stats(1)  # Last hour

def get_daily_usage_stats() -> Dict[str, Any]:
    """Get daily usage statistics."""
    return get_monitor().get_usage_stats(24)

def get_recent_violations() -> Dict[str, Any]:
    """Get recent violations report."""
    return get_monitor().get_violation_report(1)  # Last hour

def check_for_abuse() -> List[Dict[str, Any]]:
    """Check for potential abuse patterns."""
    return get_monitor().detect_abuse_patterns()