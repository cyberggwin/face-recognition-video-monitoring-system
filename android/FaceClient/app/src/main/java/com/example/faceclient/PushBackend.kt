package com.example.faceclient

import android.content.Context
import android.net.Uri
import com.google.firebase.messaging.FirebaseMessaging
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject

private val JSON_MEDIA = "application/json; charset=utf-8".toMediaType()

object PushBackend {
    fun registerCurrentToken(context: Context, client: OkHttpClient) {
        FirebaseMessaging.getInstance().token.addOnSuccessListener { token ->
            registerToken(context, client, token)
        }
    }

    fun registerToken(context: Context, client: OkHttpClient, fcmToken: String) {
        val prefs = context.getSharedPreferences("faceclient", Context.MODE_PRIVATE)
        val base = prefs.getString("serverUrl", "")?.trim()?.trimEnd('/') ?: ""
        if (base.isEmpty()) return

        val authToken = prefs.getString("token", "")?.trim().orEmpty()
        val qs = if (authToken.isNotEmpty()) "?token=${Uri.encode(authToken)}" else ""

        val payload = JSONObject().put("fcm_token", fcmToken).toString()
        val req = Request.Builder()
            .url("$base/register_device$qs")
            .post(payload.toRequestBody(JSON_MEDIA))
            .build()

        Thread {
            runCatching {
                client.newCall(req).execute().use { _ -> }
            }
        }.start()
    }

    fun unregisterCurrentToken(context: Context, client: OkHttpClient) {
        FirebaseMessaging.getInstance().token.addOnSuccessListener { token ->
            unregisterToken(context, client, token)
        }
    }

    fun unregisterToken(context: Context, client: OkHttpClient, fcmToken: String) {
        val prefs = context.getSharedPreferences("faceclient", Context.MODE_PRIVATE)
        val base = prefs.getString("serverUrl", "")?.trim()?.trimEnd('/') ?: ""
        if (base.isEmpty()) return

        val authToken = prefs.getString("token", "")?.trim().orEmpty()
        val qs = if (authToken.isNotEmpty()) "?token=${Uri.encode(authToken)}" else ""

        val payload = JSONObject().put("fcm_token", fcmToken).toString()
        val req = Request.Builder()
            .url("$base/unregister_device$qs")
            .post(payload.toRequestBody(JSON_MEDIA))
            .build()

        Thread {
            runCatching {
                client.newCall(req).execute().use { _ -> }
            }
        }.start()
    }
}