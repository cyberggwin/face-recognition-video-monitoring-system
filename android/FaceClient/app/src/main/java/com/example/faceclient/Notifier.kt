package com.example.faceclient

import android.app.NotificationChannel
import android.app.NotificationManager
import android.content.Context
import android.os.Build
import androidx.core.app.NotificationCompat
import androidx.core.app.NotificationManagerCompat

import java.util.concurrent.ConcurrentHashMap

class Notifier(private val context: Context) {

    private val channelId = "face_events"

    companion object {
        private const val NOTIF_COOLDOWN_MS = 30_000L
        private val lastByName = ConcurrentHashMap<String, Long>()
    }

    fun ensureChannel() {
        if (Build.VERSION.SDK_INT >= 26) {
            val channel = NotificationChannel(
                channelId,
                "Face events",
                NotificationManager.IMPORTANCE_DEFAULT
            )
            val nm = context.getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
            nm.createNotificationChannel(channel)
        }
    }

    fun notifyPerson(name: String) {
        // Throttle repeated notifications (same person)
        val now = System.currentTimeMillis()
        val prev = lastByName[name]
        if (prev != null && (now - prev) < NOTIF_COOLDOWN_MS) {
            return
        }
        lastByName[name] = now

        val title = "Detected: $name"
        val text = "Verified live"

        val n = NotificationCompat.Builder(context, channelId)
            .setSmallIcon(android.R.drawable.ic_menu_camera)
            .setContentTitle(title)
            .setContentText(text)
            .setPriority(NotificationCompat.PRIORITY_DEFAULT)
            .setAutoCancel(true)
            .build()

        NotificationManagerCompat.from(context).notify((System.currentTimeMillis() % 1_000_000).toInt(), n)
    }
}
