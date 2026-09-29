package com.harbor.main_app

import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine

class MainActivity : FlutterActivity() {
    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)
        // Register the `com.harbor.main_app/platform` channel. The activity
        // doubles as the context and as the activity used for runtime
        // permission requests.
        PlatformChannelHandler(this, this)
            .attach(flutterEngine.dartExecutor.binaryMessenger)
    }
}