// Harbor update server — lifecycle helpers.
//
// Shutdown correctness is the one piece of server behaviour that cannot be
// observed through the API, so it lives here as a small, directly testable
// unit instead of being buried in bin/server.dart.

import 'dart:async';

import 'package:shelf/shelf.dart';

/// Tracks the requests currently being served so that a shutdown can wait for
/// them instead of cutting them off mid-response.
///
/// The tracker is intentionally trivial: a counter plus a completer that is
/// created on demand and completed when the counter returns to zero. It makes
/// no attempt to time out — bounding the wait is the caller's job, because only
/// the caller knows how long a shutdown may take.
class InFlightTracker {
  int _active = 0;
  Completer<void>? _drained;

  /// Number of requests that have entered the handler and not yet returned,
  /// whether they returned a response or threw.
  int get active => _active;

  /// Completes once no tracked request is in flight.
  ///
  /// Completes immediately when the tracker is idle, so a shutdown that races
  /// an empty server does not wait at all.
  Future<void> get drained {
    if (_active == 0) {
      return Future<void>.value();
    }
    return (_drained ??= Completer<void>()).future;
  }

  /// Runs [request] while it counts as in flight, releasing the count even when
  /// the request throws. Accepts [FutureOr] because shelf handlers may return
  /// either a response or a future of one.
  Future<Response> track(FutureOr<Response> Function() request) async {
    _active += 1;
    try {
      return await request();
    } finally {
      _active -= 1;
      if (_active == 0) {
        final Completer<void>? pending = _drained;
        _drained = null;
        if (pending != null && !pending.isCompleted) {
          pending.complete();
        }
      }
    }
  }

  /// shelf middleware that routes every request through [track], so the count
  /// reflects exactly what the handler pipeline sees.
  Handler wrap(Handler inner) {
    return (Request request) => track(() => inner(request));
  }
}