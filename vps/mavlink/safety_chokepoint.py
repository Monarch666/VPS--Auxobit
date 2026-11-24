"""Single MAVLink Output Safety Chokepoint for Shadow Mode.

Enforces the core principle: Shadow Mode is STRICTLY OBSERVE-ONLY.
Under no circumstances can the VPS companion computer influence the flight controller
in this mode.

SAFETY GUARDS:
1. Whitelist when mode == 'shadow':
   - HEARTBEAT (<= 1 Hz)
   - NAMED_VALUE_FLOAT (<= 2 Hz per variable, for non-intrusive GCS/log monitoring)
   - STATUSTEXT (low rate, advisory only)
2. Hard Blocked when mode == 'shadow':
   - DISTANCE_SENSOR (#132)
   - ODOMETRY (#331)
   - VISION_POSITION_ESTIMATE (#102), VISION_SPEED_ESTIMATE (#103)
   - GPS_INPUT (#232)
   - COMMAND_LONG (#76), COMMAND_INT (#75)
   - PARAM_SET (#23), PARAM_REQUEST_READ (#20)
   - SET_POSITION_TARGET_LOCAL_NED (#84), SET_POSITION_TARGET_GLOBAL_INT (#86)
3. Enforces rate limits (e.g. <= 2 Hz on NAMED_VALUE_FLOAT).
4. Records blocked vs allowed packet counts and logs any illegal send attempts.
5. Can proxy/intercept calls to mav.mav.* so legacy or nested senders cannot bypass the chokepoint.
"""

from typing import Optional, Dict, Any, Callable, Set, List
import logging
import threading
import time
from pymavlink import mavutil

logger = logging.getLogger("VPS.SafetyChokepoint")


class BlockedMessageError(RuntimeError):
    """Raised if an unhandled blocked message attempt occurs in strict mode."""
    pass


class ShadowMAVLinkSafetyChokepoint:
    """Enforces outbound MAVLink packet whitelisting and rate limits."""

    WHITELIST_MESSAGES: Set[str] = {
        "HEARTBEAT",
        "NAMED_VALUE_FLOAT",
        "STATUSTEXT",
    }

    BLOCKED_MESSAGES: Set[str] = {
        "DISTANCE_SENSOR",
        "ODOMETRY",
        "VISION_POSITION_ESTIMATE",
        "VISION_SPEED_ESTIMATE",
        "GPS_INPUT",
        "COMMAND_LONG",
        "COMMAND_INT",
        "PARAM_SET",
        "PARAM_REQUEST_READ",
        "SET_POSITION_TARGET_LOCAL_NED",
        "SET_POSITION_TARGET_GLOBAL_INT",
    }

    def __init__(
        self,
        mav_connection: Optional[mavutil.mavlink_connection] = None,
        mode: str = "shadow",
        max_named_value_rate_hz: float = 2.0,
        max_heartbeat_rate_hz: float = 1.0,
    ) -> None:
        """Initialize safety chokepoint.

        Args:
            mav_connection: Active pymavlink connection object.
            mode: Operational mode ("shadow", "flight", "handheld", "sitl").
            max_named_value_rate_hz: Maximum broadcast rate for NAMED_VALUE_FLOAT (default 2 Hz).
            max_heartbeat_rate_hz: Maximum rate for HEARTBEAT packets (default 1 Hz).
        """
        self.mav = mav_connection
        self.mode = str(mode).lower()
        self.max_named_val_rate_hz = float(max_named_value_rate_hz)
        self.max_hb_rate_hz = float(max_heartbeat_rate_hz)

        self._lock = threading.RLock()
        self._allowed_counts: Dict[str, int] = {}
        self._blocked_counts: Dict[str, int] = {}
        self._last_sent_time: Dict[str, float] = {}

        if self.mav is not None and hasattr(self.mav, "mav"):
            self._install_chokepoint_proxy()

    @property
    def is_shadow_mode(self) -> bool:
        """Return True if currently operating under shadow mode constraints."""
        return self.mode == "shadow"

    def is_message_allowed(self, msg_type: str) -> bool:
        """Check if message type is permitted under current mode."""
        clean_type = msg_type.upper().strip()
        if not self.is_shadow_mode:
            # Active flight modes allow standard navigation messages
            return True
        return clean_type in self.WHITELIST_MESSAGES

    def _install_chokepoint_proxy(self) -> None:
        """Wrap mav.mav dispatch methods and raw send() with chokepoint interception."""
        mav_obj = self.mav.mav

        # 1. Intercept raw mav.send(msg) so dynamically generated or direct messages cannot bypass
        if hasattr(mav_obj, "send"):
            orig_raw_send = mav_obj.send

            def make_raw_wrapper(fn: Callable):
                def raw_send_wrapper(msg, *args, **kwargs):
                    msg_type = getattr(msg, "get_type", lambda: "")() or getattr(msg, "name", "")
                    clean_type = str(msg_type).upper().strip()
                    with self._lock:
                        if not self.is_message_allowed(clean_type):
                            self._blocked_counts[clean_type] = self._blocked_counts.get(clean_type, 0) + 1
                            logger.warning(
                                "SAFETY CHOKEPOINT HARD DROP: Prohibited '%s' packet suppressed in shadow mode (total blocked: %d)",
                                clean_type, self._blocked_counts[clean_type]
                            )
                            return False
                    return fn(msg, *args, **kwargs)
                return raw_send_wrapper

            mav_obj.send = make_raw_wrapper(orig_raw_send)

        # 2. Intercept specific high-level *_send methods to drop early before encoding
        intercept_map = {
            "distance_sensor_send": "DISTANCE_SENSOR",
            "odometry_send": "ODOMETRY",
            "vision_position_estimate_send": "VISION_POSITION_ESTIMATE",
            "vision_speed_estimate_send": "VISION_SPEED_ESTIMATE",
            "gps_input_send": "GPS_INPUT",
            "command_long_send": "COMMAND_LONG",
            "command_int_send": "COMMAND_INT",
            "param_set_send": "PARAM_SET",
            "set_position_target_local_ned_send": "SET_POSITION_TARGET_LOCAL_NED",
            "set_position_target_global_int_send": "SET_POSITION_TARGET_GLOBAL_INT",
        }

        for method_name, msg_type in intercept_map.items():
            if hasattr(mav_obj, method_name):
                orig_fn = getattr(mav_obj, method_name)

                def make_wrapper(m_type: str, fn: Callable):
                    def wrapper(*args, **kwargs):
                        return self.send_intercepted(m_type, fn, *args, **kwargs)
                    return wrapper

                setattr(mav_obj, method_name, make_wrapper(msg_type, orig_fn))

    def send_intercepted(self, msg_type: str, send_fn: Callable, *args, **kwargs) -> bool:
        """Route generic MAVLink send call through safety filter.

        Args:
            msg_type: MAVLink message name (e.g. 'DISTANCE_SENSOR', 'ODOMETRY').
            send_fn: Actual transmission callable.
            args, kwargs: Arguments to send_fn.

        Returns:
            True if packet was permitted and dispatched, False if blocked or rate-limited.
        """
        clean_type = msg_type.upper().strip()

        with self._lock:
            if not self.is_message_allowed(clean_type):
                self._blocked_counts[clean_type] = self._blocked_counts.get(clean_type, 0) + 1
                logger.warning(
                    "SAFETY CHOKEPOINT BLOCKED: Outbound '%s' suppressed in shadow mode (total blocked: %d)",
                    clean_type, self._blocked_counts[clean_type]
                )
                return False

            # Rate limit check for whitelisted messages
            t_now = time.monotonic()
            if clean_type == "NAMED_VALUE_FLOAT":
                min_interval = 1.0 / max(0.1, self.max_named_val_rate_hz)
                raw_name = args[1] if len(args) > 1 else kwargs.get("name", "")
                if isinstance(raw_name, bytes):
                    name_str = raw_name.decode("ascii", errors="ignore").rstrip("\x00")
                else:
                    name_str = str(raw_name).rstrip("\x00")
                name_key = f"NVF_{name_str}"
                last_t = self._last_sent_time.get(name_key, 0.0)
                if t_now - last_t < min_interval:
                    return False
                self._last_sent_time[name_key] = t_now

            elif clean_type == "HEARTBEAT":
                min_interval = 1.0 / max(0.1, self.max_hb_rate_hz)
                last_t = self._last_sent_time.get("HEARTBEAT", 0.0)
                if t_now - last_t < min_interval:
                    return False
                self._last_sent_time["HEARTBEAT"] = t_now

            # Execute permitted transmission
            try:
                send_fn(*args, **kwargs)
                self._allowed_counts[clean_type] = self._allowed_counts.get(clean_type, 0) + 1
                return True
            except Exception as e:
                logger.error("Failed to send whitelisted '%s': %s", clean_type, e)
                return False

    def send_heartbeat(
        self,
        type_id: int = mavutil.mavlink.MAV_TYPE_ONBOARD_CONTROLLER,
        autopilot_id: int = mavutil.mavlink.MAV_AUTOPILOT_INVALID,
        base_mode: int = 0,
        custom_mode: int = 0,
        system_status: int = mavutil.mavlink.MAV_STATE_ACTIVE,
    ) -> bool:
        """Send permitted HEARTBEAT packet subject to rate limiting."""
        if self.mav is None:
            return False

        return self.send_intercepted(
            "HEARTBEAT",
            self.mav.mav.heartbeat_send,
            type_id,
            autopilot_id,
            base_mode,
            custom_mode,
            system_status,
        )

    def send_named_value_float(
        self,
        name: str,
        value: float,
        time_boot_ms: int = 0,
    ) -> bool:
        """Send non-intrusive diagnostic float to GCS and ArduPilot DataFlash (NVF log).

        Rate-limited to max_named_value_rate_hz (default <= 2 Hz).
        """
        if self.mav is None:
            return False

        # Name must be <= 10 characters for MAVLink NAMED_VALUE_FLOAT
        clean_name = name[:10].encode("ascii", errors="ignore")

        return self.send_intercepted(
            "NAMED_VALUE_FLOAT",
            self.mav.mav.named_value_float_send,
            time_boot_ms,
            clean_name,
            float(value),
        )

    def send_statustext(
        self,
        text: str,
        severity: int = mavutil.mavlink.MAV_SEVERITY_INFO,
    ) -> bool:
        """Send human-readable diagnostic text to ground control station."""
        if self.mav is None:
            return False

        clean_text = text[:50].encode("ascii", errors="ignore")

        return self.send_intercepted(
            "STATUSTEXT",
            self.mav.mav.statustext_send,
            severity,
            clean_text,
        )

    @property
    def total_blocked_packets(self) -> int:
        """Return total count of prohibited packets dropped by the chokepoint."""
        with self._lock:
            return sum(self._blocked_counts.values())

    @property
    def total_allowed_packets(self) -> int:
        """Return total count of permitted packets dispatched through the chokepoint."""
        with self._lock:
            return sum(self._allowed_counts.values())

    def get_statistics(self) -> Dict[str, Any]:
        """Return audit snapshot of chokepoint activity."""
        with self._lock:
            return {
                "mode": self.mode,
                "is_shadow_mode": self.is_shadow_mode,
                "total_allowed": sum(self._allowed_counts.values()),
                "total_blocked": sum(self._blocked_counts.values()),
                "allowed_by_type": dict(self._allowed_counts),
                "blocked_by_type": dict(self._blocked_counts),
            }
