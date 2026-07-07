package com.example.faceclient

import com.google.firebase.messaging.FirebaseMessagingService
import com.google.firebase.messaging.RemoteMessage
import okhttp3.OkHttpClient

class PushMessagingService : FirebaseMessagingService() {
    private val notifier by lazy { Notifier(this) }
    private val client by lazy { OkHttpClient() }

    override fun onNewToken(token: String) {
        super.onNewToken(token)
        PushBackend.registerToken(this, client, token)
    }

    override fun onMessageReceived(message: RemoteMessage) {
        super.onMessageReceived(message)
        notifier.ensureChannel()

        val nameFromData = message.data["name"]?.trim().orEmpty()
        val name = if (nameFromData.isNotEmpty()) {
            nameFromData
        } else {
            "persoana"
        }
        notifier.notifyPerson(name)
    }
}