package com.harbor.main_app

import android.annotation.SuppressLint
import android.app.Activity
import android.app.NotificationChannel
import android.app.NotificationManager
import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.content.pm.ApplicationInfo
import android.content.pm.PackageManager
import android.net.ConnectivityManager
import android.net.NetworkCapabilities
import android.net.Uri
import android.os.BatteryManager
import android.os.Build
import android.os.Environment
import android.os.Handler
import android.os.Looper
import android.os.PowerManager
import android.os.StatFs
import android.os.VibrationEffect
import android.os.Vibrator
import android.os.VibratorManager
import android.text.format.DateFormat
import android.view.WindowManager
import android.widget.Toast
import androidx.core.app.ActivityCompat
import androidx.core.app.NotificationCompat
import androidx.core.app.NotificationManagerCompat
import androidx.core.content.ContextCompat
import io.flutter.plugin.common.BinaryMessenger
import io.flutter.plugin.common.MethodCall
import io.flutter.plugin.common.MethodChannel
import java.io.File
import java.util.Locale
import java.util.TimeZone

/**
 * Native half of the `com.harbor.main_app/platform` method channel.
 *
 * The channel exposes exactly one entry point, `call`, whose argument is a map
 * with a `method` string (and an optional `args` map). Routing everything
 * through one entry point keeps the Dart contract in a single table and lets
 * the Kotlin dispatcher be audited method by method.
 *
 * Failure policy: an unknown method yields
 * [MethodChannel.Result.error] with code `unsupported_method`; every other
 * exception yields code `platform_error` with the exception message. All
 * device APIs are best-effort — a single unavailable service must never crash
 * the app, so each lookup is defensive and every method body runs inside the
 * dispatcher's try/catch (or, for main-thread work, inside the posted block's
 * try/catch).
 *
 * Threading: Flutter dispatches `onMethodCall` on the platform (main) thread.
 * The methods that genuinely touch the UI — toast and clipboard access — are
 * nevertheless re-posted to the main looper so the requirement is explicit and
 * they remain safe even if the handler is ever invoked from another thread.
 */
@SuppressLint("MissingPermission")
class PlatformChannelHandler(
    private val context: Context,
    private val activity: Activity?,
) : MethodChannel.MethodCallHandler {

    companion object {
        /** Channel name shared with `platform_bridge.dart`. */
        const val CHANNEL_NAME = "com.harbor.main_app/platform"

        private const val METHOD_KEY = "method"
        private const val ARGS_KEY = "args"
        private const val NOTIFICATION_CHANNEL_ID = "harbor_notifications"
        private const val NOTIFICATION_REQUEST_CODE = 0x4841
    }

    /** Registers this handler on [messenger]. */
    fun attach(messenger: BinaryMessenger) {
        MethodChannel(messenger, CHANNEL_NAME).setMethodCallHandler(this)
    }

    override fun onMethodCall(call: MethodCall, result: MethodChannel.Result) {
        if (call.method != "call") {
            result.error(
                "unsupported_method",
                "unknown channel method: ${call.method}",
                null,
            )
            return
        }

        val method = call.argument<String>(METHOD_KEY)
        if (method == null) {
            result.error("unsupported_method", "missing '$METHOD_KEY' argument", null)
            return
        }

        @Suppress("UNCHECKED_CAST")
        val args: Map<String, Any?> =
            (call.argument<Any?>(ARGS_KEY) as? Map<String, Any?>) ?: emptyMap()

        try {
            when (method) {
                // ---- device.* -------------------------------------------------
                "device.info" -> result.success(deviceInfo())
                "device.battery" -> result.success(deviceBattery())
                "device.storage" -> result.success(deviceStorage())
                "device.connectivity" -> result.success(deviceConnectivity())
                "device.display" -> result.success(deviceDisplay())
                "device.thermal" -> result.success(deviceThermal())
                "device.locale" -> result.success(deviceLocale())

                "device.clipboard.read" -> clipboardRead(result)
                "device.clipboard.write" -> clipboardWrite(args.string("text"), result)

                "device.notify" -> deviceNotify(
                    args.int("id"),
                    args.string("title"),
                    args.string("body"),
                    args.bool("ongoing"),
                    result,
                )
                "device.vibrate" -> result.success(deviceVibrate(args.int("millis")))
                "device.toast" -> deviceToast(args.string("text"), result)
                "device.share" -> deviceShare(
                    args.string("text"),
                    args.optionalString("subject"),
                    result,
                )
                "device.installedApps" ->
                    result.success(deviceInstalledApps(args.bool("includeSystem")))
                "device.openUrl" -> deviceOpenUrl(args.string("url"), result)
                "device.openApp" -> deviceOpenApp(args.string("package"), result)

                // ---- notifications.* ------------------------------------------
                "notifications.permission" -> result.success(notificationsPermission())
                "notifications.request" -> notificationsRequest(result)

                // ---- training.* (foreground service) --------------------------
                "training.start" -> result.success(
                    trainingStart(args.string("title"), args.string("text")),
                )
                "training.update" -> result.success(
                    trainingUpdate(
                        args.int("progress"),
                        args.int("max"),
                        args.string("text"),
                    ),
                )
                "training.stop" -> result.success(trainingStop())

                else -> result.error(
                    "unsupported_method",
                    "unsupported method: $method",
                    null,
                )
            }
        } catch (e: Exception) {
            // One broken device API must not take down the caller: report it as
            // a typed platform error instead.
            result.error("platform_error", e.message ?: e.toString(), null)
        }
    }

    // ---------------------------------------------------------------------------
    // device.info / battery / storage / connectivity / display / thermal / locale
    // ---------------------------------------------------------------------------

    private fun deviceInfo(): Map<String, Any?> = mapOf(
        "manufacturer" to Build.MANUFACTURER,
        "model" to Build.MODEL,
        "brand" to Build.BRAND,
        "device" to Build.DEVICE,
        "hardware" to Build.HARDWARE,
        "supportedAbis" to Build.SUPPORTED_ABIS.toList(),
        "androidSdk" to Build.VERSION.SDK_INT,
        "androidRelease" to Build.VERSION.RELEASE,
        "isEmulator" to isEmulator(),
        "locale" to Locale.getDefault().toString(),
        "timeZone" to TimeZone.getDefault().id,
    )

    /** Standard "looks like an emulator" heuristic across every build field. */
    private fun isEmulator(): Boolean {
        val fingerprint = Build.FINGERPRINT.lowercase(Locale.US)
        val model = Build.MODEL.lowercase(Locale.US)
        val manufacturer = Build.MANUFACTURER.lowercase(Locale.US)
        val brand = Build.BRAND.lowercase(Locale.US)
        val device = Build.DEVICE.lowercase(Locale.US)
        val product = Build.PRODUCT.lowercase(Locale.US)
        val hardware = Build.HARDWARE.lowercase(Locale.US)
        return fingerprint.startsWith("generic") ||
            fingerprint.contains("emulator") ||
            model.contains("google_sdk") ||
            model.contains("emulator") ||
            model.contains("android sdk built for") ||
            manufacturer.contains("genymotion") ||
            brand.startsWith("generic") ||
            device.contains("generic") ||
            product.contains("sdk") ||
            product.contains("emulator") ||
            product.contains("simulator") ||
            hardware.contains("goldfish") ||
            hardware.contains("ranchu")
    }

    /**
     * Reads the sticky [Intent.ACTION_BATTERY_CHANGED] broadcast.
     *
     * A null sticky intent means the platform genuinely did not publish one, so
     * that is the only case reported as `platform_error`; every extra is read
     * with a default so a partially populated intent still returns a full map.
     *
     * `registerReceiver` passes a null receiver purely to read the sticky value.
     * Android 14's exported/not-exported flag requirement applies to receivers
     * registered for non-system broadcasts; `ACTION_BATTERY_CHANGED` is a
     * protected system broadcast, so it is exempt.
     */
    private fun deviceBattery(): Map<String, Any?> {
        val batteryIntent: Intent = context.registerReceiver(
            null,
            IntentFilter(Intent.ACTION_BATTERY_CHANGED),
        ) ?: throw IllegalStateException("battery status is unavailable on this device")

        val level = batteryIntent.getIntExtra(BatteryManager.EXTRA_LEVEL, -1)
        val scale = batteryIntent.getIntExtra(BatteryManager.EXTRA_SCALE, -1)
        val status = batteryIntent.getIntExtra(
            BatteryManager.EXTRA_STATUS,
            BatteryManager.BATTERY_STATUS_UNKNOWN,
        )
        val plugged = batteryIntent.getIntExtra(BatteryManager.EXTRA_PLUGGED, 0)
        val temperature = batteryIntent.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, 0)
        val health = batteryIntent.getIntExtra(
            BatteryManager.EXTRA_HEALTH,
            BatteryManager.BATTERY_HEALTH_UNKNOWN,
        )

        val fraction =
            if (scale > 0 && level >= 0) level.toDouble() / scale.toDouble() else 0.0

        return mapOf(
            "level" to fraction.coerceIn(0.0, 1.0),
            "isCharging" to (
                status == BatteryManager.BATTERY_STATUS_CHARGING ||
                    status == BatteryManager.BATTERY_STATUS_FULL
                ),
            "isFull" to (status == BatteryManager.BATTERY_STATUS_FULL),
            // EXTRA_TEMPERATURE is in tenths of a degree Celsius.
            "temperatureC" to temperature.toDouble() / 10.0,
            "health" to batteryHealthName(health),
            "plugType" to batteryPlugName(plugged),
            "status" to batteryStatusName(status),
        )
    }

    private fun batteryStatusName(status: Int): String = when (status) {
        BatteryManager.BATTERY_STATUS_CHARGING -> "charging"
        BatteryManager.BATTERY_STATUS_DISCHARGING -> "discharging"
        BatteryManager.BATTERY_STATUS_NOT_CHARGING -> "not_charging"
        BatteryManager.BATTERY_STATUS_FULL -> "full"
        else -> "unknown"
    }

    private fun batteryHealthName(health: Int): String = when (health) {
        BatteryManager.BATTERY_HEALTH_GOOD -> "good"
        BatteryManager.BATTERY_HEALTH_OVERHEAT -> "overheat"
        BatteryManager.BATTERY_HEALTH_DEAD -> "dead"
        BatteryManager.BATTERY_HEALTH_OVER_VOLTAGE -> "over_voltage"
        BatteryManager.BATTERY_HEALTH_UNSPECIFIED_FAILURE -> "unspecified_failure"
        BatteryManager.BATTERY_HEALTH_COLD -> "cold"
        else -> "unknown"
    }

    private fun batteryPlugName(plugged: Int): String = when (plugged) {
        BatteryManager.BATTERY_PLUGGED_AC -> "ac"
        BatteryManager.BATTERY_PLUGGED_USB -> "usb"
        BatteryManager.BATTERY_PLUGGED_WIRELESS -> "wireless"
        BatteryManager.BATTERY_PLUGGED_DOCK -> "dock"
        else -> "none"
    }

    /**
     * Internal/external volume sizes plus this app's own footprint.
     *
     * Byte counts are returned as Kotlin `Long`: the standard message codec
     * decodes both `Int` and `Long` into a Dart `int`, and a modern /data
     * partition is far larger than 32 bits, so widening the reply is the only
     * way to report the real number without overflow.
     */
    @Suppress("DEPRECATION")
    private fun deviceStorage(): Map<String, Any?> {
        val dataDir = Environment.getDataDirectory()
        val internal = StatFs(dataDir.path)
        val internalFree = internal.availableBytes
        val internalTotal = internal.totalBytes
        val appUsed = directorySizeBytes(context.filesDir)

        var externalFree = 0L
        var externalTotal = 0L
        try {
            val externalDir: File? = Environment.getExternalStorageDirectory()
            if (externalDir != null && externalDir.exists()) {
                val external = StatFs(externalDir.path)
                externalFree = external.availableBytes
                externalTotal = external.totalBytes
            }
        } catch (e: Exception) {
            // A missing or unmounted external volume is not an error: report
            // zero rather than letting the call fail.
            externalFree = 0L
            externalTotal = 0L
        }

        return mapOf(
            "internalFreeBytes" to internalFree,
            "internalTotalBytes" to internalTotal,
            "appUsedBytes" to appUsed,
            "externalFreeBytes" to externalFree,
            "externalTotalBytes" to externalTotal,
        )
    }

    /** Sums the file lengths under [root], tolerating unreadable entries. */
    private fun directorySizeBytes(root: File): Long {
        var total = 0L
        val pending = ArrayDeque<File>()
        pending.addLast(root)
        while (pending.isNotEmpty()) {
            val current = pending.removeLast()
            val children = current.listFiles() ?: continue
            for (child in children) {
                if (child.isDirectory) {
                    pending.addLast(child)
                } else {
                    total += child.length()
                }
            }
        }
        return total
    }

    /**
     * Reads the active network's transports and capabilities.
     *
     * `activeNetwork` exists only from API 23; below that the deprecated
     * `activeNetworkInfo` snapshot is used so API 21/22 still report correctly.
     */
    @Suppress("DEPRECATION")
    private fun deviceConnectivity(): Map<String, Any?> {
        val connectivityManager =
            context.getSystemService(Context.CONNECTIVITY_SERVICE) as? ConnectivityManager
                ?: return noNetwork()

        val network = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            connectivityManager.activeNetwork
        } else {
            null
        }
        val capabilities = network?.let { connectivityManager.getNetworkCapabilities(it) }

        if (capabilities != null) {
            val type = when {
                capabilities.hasTransport(NetworkCapabilities.TRANSPORT_WIFI) -> "wifi"
                capabilities.hasTransport(NetworkCapabilities.TRANSPORT_CELLULAR) -> "cellular"
                capabilities.hasTransport(NetworkCapabilities.TRANSPORT_ETHERNET) -> "ethernet"
                else -> "other"
            }
            return mapOf(
                "connected" to capabilities.hasCapability(
                    NetworkCapabilities.NET_CAPABILITY_INTERNET,
                ),
                "type" to type,
                "isMetered" to connectivityManager.isActiveNetworkMetered,
                "isVpn" to capabilities.hasTransport(NetworkCapabilities.TRANSPORT_VPN),
            )
        }

        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.M) {
            val legacy = connectivityManager.activeNetworkInfo
            if (legacy != null && legacy.isConnected) {
                val type = when (legacy.type) {
                    ConnectivityManager.TYPE_WIFI -> "wifi"
                    ConnectivityManager.TYPE_MOBILE -> "cellular"
                    ConnectivityManager.TYPE_ETHERNET -> "ethernet"
                    else -> "other"
                }
                return mapOf(
                    "connected" to true,
                    "type" to type,
                    "isMetered" to connectivityManager.isActiveNetworkMetered,
                    "isVpn" to (legacy.type == ConnectivityManager.TYPE_VPN),
                )
            }
        }

        return noNetwork()
    }

    private fun noNetwork(): Map<String, Any?> = mapOf(
        "connected" to false,
        "type" to "none",
        "isMetered" to false,
        "isVpn" to false,
    )

    /**
     * Window metrics. `defaultDisplay` is deprecated on API 30+ but still
     * returns the display for this window on every API level, whereas
     * `Context.getDisplay()` needs a visual context and would throw for an
     * application context.
     */
    @Suppress("DEPRECATION")
    private fun deviceDisplay(): Map<String, Any?> {
        val metrics = context.resources.displayMetrics
        val windowManager = context.getSystemService(Context.WINDOW_SERVICE) as? WindowManager
        val refreshRate = windowManager?.defaultDisplay?.refreshRate?.toDouble() ?: 0.0
        return mapOf(
            "widthPx" to metrics.widthPixels,
            "heightPx" to metrics.heightPixels,
            "density" to metrics.density.toDouble(),
            "refreshRateHz" to refreshRate,
            "fontScale" to context.resources.configuration.fontScale.toDouble(),
        )
    }

    /** Thermal status mapping; `currentThermalStatus` itself is API 29+. */
    private fun deviceThermal(): Map<String, Any?> {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.Q) {
            return mapOf("status" to "unknown")
        }
        val powerManager =
            context.getSystemService(Context.POWER_SERVICE) as? PowerManager
                ?: return mapOf("status" to "unknown")
        val status = when (powerManager.currentThermalStatus) {
            PowerManager.THERMAL_STATUS_NONE -> "none"
            PowerManager.THERMAL_STATUS_LIGHT -> "light"
            PowerManager.THERMAL_STATUS_MODERATE -> "moderate"
            PowerManager.THERMAL_STATUS_SEVERE -> "severe"
            PowerManager.THERMAL_STATUS_CRITICAL -> "critical"
            PowerManager.THERMAL_STATUS_EMERGENCY -> "emergency"
            PowerManager.THERMAL_STATUS_SHUTDOWN -> "shutdown"
            else -> "unknown"
        }
        return mapOf("status" to status)
    }

    private fun deviceLocale(): Map<String, Any?> {
        val locale = Locale.getDefault()
        return mapOf(
            "language" to locale.language,
            "country" to locale.country,
            "timeZone" to TimeZone.getDefault().id,
            "uses24Hour" to DateFormat.is24HourFormat(context),
        )
    }

    // ---------------------------------------------------------------------------
    // Clipboard / notification / vibration / toast / share / apps / intents
    // ---------------------------------------------------------------------------

    private fun clipboardRead(result: MethodChannel.Result) {
        runOnMainThread(result) {
            val manager =
                context.getSystemService(Context.CLIPBOARD_SERVICE) as? ClipboardManager
            val clip = manager?.primaryClip
            val text = if (clip != null && clip.itemCount > 0) {
                clip.getItemAt(0).coerceToText(context).toString()
            } else {
                ""
            }
            result.success(mapOf("text" to text))
        }
    }

    private fun clipboardWrite(text: String, result: MethodChannel.Result) {
        runOnMainThread(result) {
            val manager =
                context.getSystemService(Context.CLIPBOARD_SERVICE) as? ClipboardManager
                    ?: throw IllegalStateException("clipboard service is unavailable")
            manager.setPrimaryClip(ClipData.newPlainText("Harbor", text))
            result.success(mapOf("ok" to true))
        }
    }

    private fun deviceNotify(
        id: Int,
        title: String,
        body: String,
        ongoing: Boolean,
        result: MethodChannel.Result,
    ) {
        runOnMainThread(result) {
            ensureNotificationChannel(
                NOTIFICATION_CHANNEL_ID,
                context.getString(R.string.harbor_notification_channel_name),
                context.getString(R.string.harbor_notification_channel_description),
                NotificationManager.IMPORTANCE_DEFAULT,
            )
            val notification = NotificationCompat.Builder(context, NOTIFICATION_CHANNEL_ID)
                .setSmallIcon(android.R.drawable.stat_sys_download)
                .setContentTitle(title)
                .setContentText(body)
                .setStyle(NotificationCompat.BigTextStyle().bigText(body))
                .setPriority(NotificationCompat.PRIORITY_DEFAULT)
                .setAutoCancel(!ongoing)
                .setOngoing(ongoing)
                .build()
            NotificationManagerCompat.from(context).notify(id, notification)
            result.success(mapOf("ok" to true))
        }
    }

    /**
     * Vibrates through [VibratorManager] on API 31+ and the legacy [Vibrator]
     * service below it; [VibrationEffect] itself needs API 26, so older devices
     * fall back to the deprecated duration-only call.
     */
    @Suppress("DEPRECATION")
    private fun deviceVibrate(millis: Int): Map<String, Any?> {
        val durationMillis = millis.coerceAtLeast(0).toLong()
        val vibrator: Vibrator? = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            context.getSystemService(VibratorManager::class.java)?.defaultVibrator
        } else {
            context.getSystemService(Context.VIBRATOR_SERVICE) as? Vibrator
        }
        if (vibrator == null || !vibrator.hasVibrator()) {
            return mapOf("ok" to false)
        }
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            vibrator.vibrate(
                VibrationEffect.createOneShot(
                    durationMillis,
                    VibrationEffect.DEFAULT_AMPLITUDE,
                ),
            )
        } else {
            vibrator.vibrate(durationMillis)
        }
        return mapOf("ok" to true)
    }

    private fun deviceToast(text: String, result: MethodChannel.Result) {
        runOnMainThread(result) {
            Toast.makeText(context, text, Toast.LENGTH_SHORT).show()
            result.success(mapOf("ok" to true))
        }
    }

    private fun deviceShare(
        text: String,
        subject: String?,
        result: MethodChannel.Result,
    ) {
        runOnMainThread(result) {
            val sendIntent = Intent(Intent.ACTION_SEND).apply {
                type = "text/plain"
                putExtra(Intent.EXTRA_TEXT, text)
                if (subject != null) {
                    putExtra(Intent.EXTRA_SUBJECT, subject)
                }
            }
            val chooser = Intent.createChooser(sendIntent, null).apply {
                addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
            }
            context.startActivity(chooser)
            result.success(mapOf("ok" to true))
        }
    }

    /**
     * Lists launcher activities.
     *
     * The deprecated two-argument `queryIntentActivities` is used deliberately:
     * the newer `PackageInfoFlags` overload only exists from API 33 and would
     * need a version branch, whereas this form compiles and runs on every level
     * from minSdk 21. Because no `QUERY_ALL_PACKAGES` permission is declared,
     * Android 11+ only returns packages visible through the manifest `<queries>`
     * block, which is exactly the intended surface.
     */
    @Suppress("DEPRECATION")
    private fun deviceInstalledApps(includeSystem: Boolean): Map<String, Any?> {
        val launcherIntent = Intent(Intent.ACTION_MAIN)
            .addCategory(Intent.CATEGORY_LAUNCHER)
        val resolved = context.packageManager.queryIntentActivities(launcherIntent, 0)
        val packages = LinkedHashSet<String>()
        for (resolveInfo in resolved) {
            val activityInfo = resolveInfo.activityInfo ?: continue
            val applicationInfo = activityInfo.applicationInfo ?: continue
            val isSystem =
                (applicationInfo.flags and ApplicationInfo.FLAG_SYSTEM) != 0
            if (!includeSystem && isSystem) {
                continue
            }
            packages.add(activityInfo.packageName)
        }
        val sorted = packages.sorted()
        return mapOf(
            "count" to sorted.size,
            "packages" to sorted,
        )
    }

    private fun deviceOpenUrl(url: String, result: MethodChannel.Result) {
        runOnMainThread(result) {
            val intent = Intent(Intent.ACTION_VIEW, Uri.parse(url)).apply {
                addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
            }
            context.startActivity(intent)
            result.success(mapOf("ok" to true))
        }
    }

    private fun deviceOpenApp(packageName: String, result: MethodChannel.Result) {
        runOnMainThread(result) {
            val launchIntent =
                context.packageManager.getLaunchIntentForPackage(packageName)
            if (launchIntent == null) {
                result.error(
                    "platform_error",
                    "no launch intent for $packageName",
                    null,
                )
                return@runOnMainThread
            }
            launchIntent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
            context.startActivity(launchIntent)
            result.success(mapOf("ok" to true))
        }
    }

    // ---------------------------------------------------------------------------
    // notifications.*
    // ---------------------------------------------------------------------------

    private fun isNotificationPermissionGranted(): Boolean =
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            ContextCompat.checkSelfPermission(
                context,
                android.Manifest.permission.POST_NOTIFICATIONS,
            ) == PackageManager.PERMISSION_GRANTED
        } else {
            true
        }

    private fun notificationsPermission(): Map<String, Any?> = mapOf(
        "granted" to isNotificationPermissionGranted(),
        "sdkInt" to Build.VERSION.SDK_INT,
    )

    /**
     * Requests `POST_NOTIFICATIONS` and replies immediately.
     *
     * `requestPermissions` is asynchronous: the user may take seconds to answer,
     * and a channel `Result` may only be completed once, so this cannot await
     * the dialog. It returns the grant state *at the moment of the request*;
     * callers must re-query `notifications.permission` after the dialog closes
     * to learn the user's actual choice. Below API 33 there is nothing to
     * request and the answer is always `true`.
     */
    private fun notificationsRequest(result: MethodChannel.Result) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            val currentActivity = activity
            if (currentActivity != null) {
                ActivityCompat.requestPermissions(
                    currentActivity,
                    arrayOf(android.Manifest.permission.POST_NOTIFICATIONS),
                    NOTIFICATION_REQUEST_CODE,
                )
            }
        }
        result.success(mapOf("granted" to isNotificationPermissionGranted()))
    }

    // ---------------------------------------------------------------------------
    // training.*
    // ---------------------------------------------------------------------------

    private fun trainingStart(title: String, text: String): Map<String, Any?> {
        val intent = Intent(context, TrainingService::class.java).apply {
            action = TrainingService.ACTION_START
            putExtra(TrainingService.EXTRA_TITLE, title)
            putExtra(TrainingService.EXTRA_TEXT, text)
        }
        ContextCompat.startForegroundService(context, intent)
        return mapOf("ok" to true)
    }

    private fun trainingUpdate(
        progress: Int,
        max: Int,
        text: String,
    ): Map<String, Any?> {
        val intent = Intent(context, TrainingService::class.java).apply {
            action = TrainingService.ACTION_UPDATE
            putExtra(TrainingService.EXTRA_PROGRESS, progress)
            putExtra(TrainingService.EXTRA_MAX, max)
            putExtra(TrainingService.EXTRA_TEXT, text)
        }
        ContextCompat.startForegroundService(context, intent)
        return mapOf("ok" to true)
    }

    private fun trainingStop(): Map<String, Any?> {
        val intent = Intent(context, TrainingService::class.java).apply {
            action = TrainingService.ACTION_STOP
        }
        context.startService(intent)
        return mapOf("ok" to true)
    }

    // ---------------------------------------------------------------------------
    // Helpers
    // ---------------------------------------------------------------------------

    /**
     * Creates the channel used by `device.notify` when missing. No-op below
     * API 26, where notification channels do not exist.
     */
    private fun ensureNotificationChannel(
        id: String,
        name: String,
        description: String,
        importance: Int,
    ) {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.O) {
            return
        }
        val manager =
            context.getSystemService(Context.NOTIFICATION_SERVICE) as? NotificationManager
                ?: return
        if (manager.getNotificationChannel(id) != null) {
            return
        }
        val channel = NotificationChannel(id, name, importance)
        channel.description = description
        manager.createNotificationChannel(channel)
    }

    /**
     * Posts [block] to the main looper and completes [result] from there.
     * Errors raised inside the block are reported as `platform_error`, since
     * the dispatcher's own try/catch has already returned by then.
     */
    private fun runOnMainThread(result: MethodChannel.Result, block: () -> Unit) {
        Handler(Looper.getMainLooper()).post {
            try {
                block()
            } catch (e: Exception) {
                result.error("platform_error", e.message ?: e.toString(), null)
            }
        }
    }

    private fun Map<String, Any?>.string(key: String): String =
        this[key]?.toString()
            ?: throw IllegalArgumentException("missing string argument '$key'")

    private fun Map<String, Any?>.optionalString(key: String): String? =
        this[key]?.toString()

    private fun Map<String, Any?>.int(key: String, default: Int = 0): Int {
        val value = this[key] ?: return default
        return when (value) {
            is Int -> value
            is Long -> value.toInt()
            is Double -> value.toInt()
            is Number -> value.toInt()
            is String -> value.toIntOrNull()
                ?: throw IllegalArgumentException("argument '$key' is not an integer")
            else -> throw IllegalArgumentException("argument '$key' is not an integer")
        }
    }

    private fun Map<String, Any?>.bool(key: String, default: Boolean = false): Boolean {
        val value = this[key] ?: return default
        return when (value) {
            is Boolean -> value
            is Number -> value.toInt() != 0
            is String -> value.equals("true", ignoreCase = true) || value == "1"
            else -> throw IllegalArgumentException("argument '$key' is not a boolean")
        }
    }
}