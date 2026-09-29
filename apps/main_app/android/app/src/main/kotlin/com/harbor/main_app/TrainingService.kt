package com.harbor.main_app

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.Build
import android.os.IBinder
import androidx.core.app.NotificationCompat

/**
 * Started, foreground service that keeps a long on-device training job visible
 * in the notification shade — and the process alive — while the app is
 * backgrounded.
 *
 * The service is driven entirely by intents from [PlatformChannelHandler]:
 * [ACTION_START] and [ACTION_UPDATE] (re)post the same ongoing notification
 * with progress, and [ACTION_STOP] removes it and shuts the service down. The
 * notification's own "Stop" action routes [ACTION_STOP] straight back here, so
 * a user can cancel training from the shade without reopening the app.
 *
 * The notification deliberately uses the framework icon
 * `android.R.drawable.stat_sys_download`: the repository ships no binary
 * drawables, and adding one just for this notification would be the first
 * raster asset in the tree.
 */
class TrainingService : Service() {

    companion object {
        const val ACTION_START = "com.harbor.main_app.action.TRAINING_START"
        const val ACTION_UPDATE = "com.harbor.main_app.action.TRAINING_UPDATE"
        const val ACTION_STOP = "com.harbor.main_app.action.TRAINING_STOP"

        const val EXTRA_TITLE = "com.harbor.main_app.extra.TRAINING_TITLE"
        const val EXTRA_TEXT = "com.harbor.main_app.extra.TRAINING_TEXT"
        const val EXTRA_PROGRESS = "com.harbor.main_app.extra.TRAINING_PROGRESS"
        const val EXTRA_MAX = "com.harbor.main_app.extra.TRAINING_MAX"

        /** Stable id so START and UPDATE replace one another's notification. */
        private const val NOTIFICATION_ID = 0x4841

        private const val CHANNEL_ID = "harbor_training"
        private const val REQUEST_CODE_CONTENT = 0x4842
        private const val REQUEST_CODE_STOP = 0x4843
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        // A sticky restart delivers a null intent: there is no job to describe,
        // so tear down rather than resurrect a stale progress card.
        if (intent == null) {
            teardown()
            return START_NOT_STICKY
        }

        return when (intent.action) {
            ACTION_START, ACTION_UPDATE -> {
                val title = intent.getStringExtra(EXTRA_TITLE)
                    ?: getString(R.string.harbor_training_default_title)
                val text = intent.getStringExtra(EXTRA_TEXT)
                    ?: getString(R.string.harbor_training_default_text)
                val max = intent.getIntExtra(EXTRA_MAX, 0)
                val progress = intent.getIntExtra(EXTRA_PROGRESS, 0)
                promoteToForeground(buildNotification(title, text, progress, max))
                START_STICKY
            }

            ACTION_STOP -> {
                teardown()
                START_NOT_STICKY
            }

            else -> {
                teardown()
                START_NOT_STICKY
            }
        }
    }

    /** Shows the notification as the foreground-service notification. */
    private fun promoteToForeground(notification: Notification) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            // targetSdk 34 requires the foreground-service type to be passed at
            // startForeground time; dataSync matches the manifest declaration
            // and FOREGROUND_SERVICE_DATA_SYNC permission.
            startForeground(
                NOTIFICATION_ID,
                notification,
                ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC,
            )
        } else {
            startForeground(NOTIFICATION_ID, notification)
        }
    }

    /** Removes the foreground notification and stops this service. */
    private fun teardown() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N) {
            stopForeground(STOP_FOREGROUND_REMOVE)
        } else {
            @Suppress("DEPRECATION")
            stopForeground(true)
        }
        stopSelf()
    }

    private fun buildNotification(
        title: String,
        text: String,
        progress: Int,
        max: Int,
    ): Notification {
        ensureChannel()

        val contentIntent = PendingIntent.getActivity(
            this,
            REQUEST_CODE_CONTENT,
            Intent(this, MainActivity::class.java).apply {
                addFlags(Intent.FLAG_ACTIVITY_SINGLE_TOP)
            },
            pendingIntentFlags(),
        )

        // The action re-enters this service through onStartCommand, where
        // ACTION_STOP tears the foreground state down.
        val stopIntent = PendingIntent.getService(
            this,
            REQUEST_CODE_STOP,
            Intent(this, TrainingService::class.java).apply {
                action = ACTION_STOP
            },
            pendingIntentFlags(),
        )

        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setSmallIcon(android.R.drawable.stat_sys_download)
            .setContentTitle(title)
            .setContentText(text)
            .setStyle(NotificationCompat.BigTextStyle().bigText(text))
            .setOnlyAlertOnce(true)
            .setOngoing(true)
            .setProgress(max, progress, false)
            .setContentIntent(contentIntent)
            .addAction(
                android.R.drawable.ic_menu_close_clear_cancel,
                getString(R.string.harbor_training_stop),
                stopIntent,
            )
            .build()
    }

    /** Creates the LOW-importance training channel once, on API 26+. */
    private fun ensureChannel() {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.O) {
            return
        }
        val manager =
            getSystemService(Context.NOTIFICATION_SERVICE) as? NotificationManager
                ?: return
        if (manager.getNotificationChannel(CHANNEL_ID) != null) {
            return
        }
        val channel = NotificationChannel(
            CHANNEL_ID,
            getString(R.string.harbor_training_channel_name),
            NotificationManager.IMPORTANCE_LOW,
        )
        channel.description = getString(R.string.harbor_training_channel_description)
        channel.setShowBadge(false)
        manager.createNotificationChannel(channel)
    }

    /**
     * `FLAG_IMMUTABLE` only exists from API 23; below that the flags are just
     * the update-current bit.
     */
    private fun pendingIntentFlags(): Int {
        val base = PendingIntent.FLAG_UPDATE_CURRENT
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            base or PendingIntent.FLAG_IMMUTABLE
        } else {
            base
        }
    }
}