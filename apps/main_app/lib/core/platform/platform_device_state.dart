// Real device conditions, read through the platform channel.
//
// The legacy engine tab keeps a *manual* source so the training gate can be
// exercised without a device. This one is the opposite: it reports what the
// phone actually says, so the idle-training gate protects the battery and the
// thermals for real rather than at the user's pleasure.
//
// One honest approximation: `isIdle` is reported as true. The gate's purpose on
// Android is to avoid draining a battery or cooking a CPU, and those are exactly
// the conditions the platform can measure — charging state and thermal status.
// Idleness is not measurable through this channel, and pretending to measure it
// (by guessing from a lifecycle event the poll cannot observe while suspended)
// would gate learning on a signal that is wrong half the time. Charging and
// thermal are the constraints that matter, and they are read exactly.

import '../engine/on_device_lora.dart';
import 'platform_bridge.dart';

/// Source of device conditions for the idle-training scheduler.
///
/// It is an interface because two very different things implement it: this file
/// reads the real battery and thermals, and the app's engine tab drives a manual
/// simulator so the gate can be exercised on a desk with no phone attached.
abstract class DeviceStateSource {
  /// Reads the current device conditions.
  Future<DeviceState> read();
}

/// A [DeviceStateSource] backed by the Android battery and thermal APIs.
class PlatformDeviceStateSource implements DeviceStateSource {
  /// Creates a source over [bridge].
  PlatformDeviceStateSource(this.bridge);

  /// The channel to read from.
  final PlatformBridge bridge;

  @override
  Future<DeviceState> read() async {
    try {
      final Map<String, Object?> battery = await bridge.battery();
      final Map<String, Object?> thermal = await bridge.thermal();
      final Object? level = battery['level'];
      return DeviceState(
        isIdle: true,
        // `isCharging` is true for both CHARGING and FULL, which is the right
        // answer for a training gate: a full battery on the charger is the best
        // possible moment to spend compute.
        isCharging: battery['isCharging'] == true,
        batteryLevel: level is num ? level.toDouble().clamp(0.0, 1.0) : 0.0,
        thermalState: _thermalState('${thermal['status'] ?? thermal['thermalStatus']}'),
      );
    } on Object {
      // An unreadable battery must fail closed: the gate then declines to run
      // rather than training on an unknown power state.
      return DeviceState.unknown;
    }
  }

  /// Translates Android's eight thermal levels onto the four the gate knows.
  ///
  /// Android distinguishes more shades of hot than a training decision needs;
  /// collapsing `critical`/`emergency`/`shutdown` into [ThermalState.critical]
  /// guarantees the gate stops at the first genuinely dangerous level rather
  /// than trusting a mapping for a state the OS may add later.
  static ThermalState _thermalState(String raw) {
    switch (raw.toLowerCase()) {
      case 'none':
      case 'light':
        return ThermalState.nominal;
      case 'moderate':
        return ThermalState.fair;
      case 'severe':
        return ThermalState.serious;
      case 'critical':
      case 'emergency':
      case 'shutdown':
        return ThermalState.critical;
      default:
        return ThermalState.nominal;
    }
  }
}