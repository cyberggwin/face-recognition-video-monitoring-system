package com.example.faceclient

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Build
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.fadeIn
import androidx.compose.animation.slideInVertically
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.spring
import androidx.compose.foundation.background
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.offset
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.MutableState
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.Response
import okhttp3.WebSocket
import okhttp3.WebSocketListener
import org.json.JSONObject
import java.text.SimpleDateFormat
import java.util.Calendar
import java.util.Date
import java.util.Locale
import java.util.concurrent.TimeUnit

private enum class UiScreen {
    WELCOME,
    DASHBOARD,
    SETTINGS,
}

class MainActivity : ComponentActivity() {

    private val okHttp by lazy {
        OkHttpClient.Builder()
            .readTimeout(0, TimeUnit.MILLISECONDS)
            .build()
    }

    private val notifier by lazy { Notifier(this) }

    private val permissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { /* no-op */ }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        notifier.ensureChannel()
        requestNotificationsIfNeeded()
        PushBackend.registerCurrentToken(this, okHttp)

        setContent {
            MaterialTheme {
                AppUI(
                    appContext = this,
                    getDefaultServerUrl = { loadServerUrl() },
                    saveServerUrl = { saveServerUrl(it) },
                    getDefaultToken = { loadToken() },
                    saveToken = { saveToken(it) },
                    getPushArmed = { loadPushArmed() },
                    savePushArmed = { savePushArmed(it) },
                    client = okHttp,
                    notifier = notifier,
                )
            }
        }
    }

    private fun requestNotificationsIfNeeded() {
        if (Build.VERSION.SDK_INT >= 33) {
            val granted = ContextCompat.checkSelfPermission(this, Manifest.permission.POST_NOTIFICATIONS) == PackageManager.PERMISSION_GRANTED
            if (!granted) {
                permissionLauncher.launch(Manifest.permission.POST_NOTIFICATIONS)
            }
        }
    }

    private fun prefs() = getSharedPreferences("faceclient", Context.MODE_PRIVATE)

    private fun loadServerUrl(): String = prefs().getString("serverUrl", "http://192.168.1.2:8000")!!

    private fun loadToken(): String = prefs().getString("token", "")!!

    private fun saveServerUrl(url: String) {
        prefs().edit().putString("serverUrl", url).apply()
    }

    private fun saveToken(token: String) {
        prefs().edit().putString("token", token).apply()
    }

    private fun loadPushArmed(): Boolean = prefs().getBoolean("pushArmed", false)

    private fun savePushArmed(v: Boolean) {
        prefs().edit().putBoolean("pushArmed", v).apply()
    }
}

@Composable
private fun AppUI(
    appContext: Context,
    getDefaultServerUrl: () -> String,
    saveServerUrl: (String) -> Unit,
    getDefaultToken: () -> String,
    saveToken: (String) -> Unit,
    getPushArmed: () -> Boolean,
    savePushArmed: (Boolean) -> Unit,
    client: OkHttpClient,
    notifier: Notifier,
) {
    val scope = rememberCoroutineScope()

    val serverUrl = remember { mutableStateOf(getDefaultServerUrl()) }
    val token = remember { mutableStateOf(getDefaultToken()) }
    val snapshotStatus = remember { mutableStateOf("disconnected") }
    val wsStatus = remember { mutableStateOf("disconnected") }
    val lastEvent = remember { mutableStateOf("-") }
    val lastEventTs = remember { mutableStateOf(0.0) }
    val bitmapState = remember { mutableStateOf<Bitmap?>(null) }
    val pushArmed = remember { mutableStateOf(getPushArmed()) }
    val recentEvents = remember { mutableStateListOf<EventItem>() }

    val initialScreen = if (pushArmed.value) UiScreen.DASHBOARD else UiScreen.WELCOME
    val screen = rememberSaveable { mutableStateOf(initialScreen) }
    val showHelp = remember { mutableStateOf(false) }
    val showHistory = remember { mutableStateOf(false) }
    val showChangeUrl = remember { mutableStateOf(false) }
    val showStatus = remember { mutableStateOf(false) }
    val timeFmt = remember { SimpleDateFormat("HH:mm:ss", Locale.getDefault()) }

    val jobs = remember { mutableStateOf<Jobs?>(null) }

    fun connect() {
        disconnect(jobs, snapshotStatus, wsStatus)

        val base = serverUrl.value.trim().trimEnd('/')
        saveServerUrl(base)

        val tokenVal = token.value.trim()
        saveToken(tokenVal)

        pushArmed.value = true
        savePushArmed(true)

        PushBackend.registerCurrentToken(appContext, client)

        val qs = if (tokenVal.isNotEmpty()) "?token=$tokenVal" else ""

        fun onPersonDetected(name: String, dist: Double, tsSec: Double) {
            if (tsSec <= lastEventTs.value) {
                return
            }
            lastEventTs.value = tsSec
            lastEvent.value = "$name | ${String.format("%.2f", dist)}"

            val tsMs = (tsSec * 1000.0).toLong()
            recentEvents.add(0, EventItem(tsMs = tsMs, name = name, dist = dist))
            if (recentEvents.size > 300) {
                recentEvents.removeRange(300, recentEvents.size)
            }

            notifier.notifyPerson(name)
        }

        val streamJob = scope.launch(Dispatchers.IO) {
            snapshotStatus.value = "connecting (snapshot loop)"
            while (true) {
                try {
                    val bmp = fetchSnapshot(client, "$base/snapshot.jpg$qs")
                    if (bmp != null) {
                        bitmapState.value = bmp
                        snapshotStatus.value = "connected"
                    }
                } catch (_: Exception) {
                    snapshotStatus.value = "snapshot error"
                }
                delay(120)
            }
        }

        val wsUrl = base.replaceFirst("http://", "ws://").replaceFirst("https://", "wss://") + "/ws" + qs
        val wsReq = Request.Builder().url(wsUrl).build()
        val ws = client.newWebSocket(wsReq, object : WebSocketListener() {
            override fun onOpen(webSocket: WebSocket, response: Response) {
                wsStatus.value = "connected"
                // send pings so server keeps receive loop alive
                scope.launch(Dispatchers.IO) {
                    while (true) {
                        webSocket.send("ping")
                        delay(15_000)
                    }
                }
            }

            override fun onMessage(webSocket: WebSocket, text: String) {
                try {
                    val obj = JSONObject(text)
                    if (obj.optString("type") == "person_detected") {
                        val name = obj.optString("name")
                        val dist = obj.optDouble("dist")
                        val ts = obj.optDouble("ts", 0.0)
                        onPersonDetected(name, dist, ts)
                    }
                } catch (_: Exception) {
                    // ignore
                }
            }

            override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
                wsStatus.value = "error: ${t.message}"
            }

            override fun onClosed(webSocket: WebSocket, code: Int, reason: String) {
                wsStatus.value = "closed"
            }
        })

        // Fallback: poll last event periodically in case WebSocket is blocked/interrupted.
        val eventPollJob = scope.launch(Dispatchers.IO) {
            while (true) {
                try {
                    val obj = fetchLastEvent(client, "$base/last_event$qs")
                    if (obj != null && obj.optString("type") == "person_detected") {
                        val ts = obj.optDouble("ts", 0.0)
                        val name = obj.optString("name")
                        val dist = obj.optDouble("dist")
                        onPersonDetected(name, dist, ts)
                    }
                } catch (_: Exception) {
                    // ignore polling errors
                }
                delay(1_000)
            }
        }

        jobs.value = Jobs(streamJob = streamJob, eventPollJob = eventPollJob, webSocket = ws)
    }

    LaunchedEffect(Unit) {
        if (pushArmed.value) {
            connect()
        }
    }

    if (showHelp.value) {
        AlertDialog(
            onDismissRequest = { showHelp.value = false },
            title = { Text("Help") },
            text = {
                Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    Text("1) Introdu adresa camerei (Server URL) si tokenul daca folosesti autentificare.")
                    Text("2) Apasa Connect pentru live stream + armare notificari push.")
                    Text("3) Poti inchide aplicatia; push ramane activ.")
                    Text("4) Disconnect opreste explicit alertele pentru telefonul tau.")
                    Text("5) Events afiseaza istoricul din ziua curenta.")
                }
            },
            confirmButton = {
                TextButton(onClick = { showHelp.value = false }) { Text("OK") }
            }
        )
    }

    if (showHistory.value) {
        AlertDialog(
            onDismissRequest = { showHistory.value = false },
            title = { Text("All Events (${recentEvents.size})") },
            text = {
                Column(
                    modifier = Modifier.height(320.dp).verticalScroll(rememberScrollState()),
                    verticalArrangement = Arrangement.spacedBy(6.dp)
                ) {
                    if (recentEvents.isEmpty()) {
                        Text("No events yet for today.")
                    } else {
                        recentEvents.forEach { ev ->
                            Text(
                                "${timeFmt.format(Date(ev.tsMs))}  |  ${ev.name}  |  ${String.format("%.2f", ev.dist)}",
                                fontFamily = FontFamily.Monospace,
                                fontSize = 12.sp
                            )
                        }
                    }
                }
            },
            confirmButton = {
                TextButton(onClick = { showHistory.value = false }) { Text("Close") }
            }
        )
    }

    when (screen.value) {
        UiScreen.WELCOME -> {
            WelcomeScreen(
                onContinue = {
                    screen.value = UiScreen.DASHBOARD
                    if (jobs.value == null) {
                        connect()
                    }
                },
                onHelp = { showHelp.value = true },
                onSettings = { screen.value = UiScreen.SETTINGS }
            )
        }

        UiScreen.DASHBOARD -> {
            DashboardScreen(
                bitmapState = bitmapState,
                recentEvents = recentEvents,
                timeFmt = timeFmt,
                onBack = { screen.value = UiScreen.WELCOME },
                onAllEvents = { showHistory.value = true }
            )
        }

        UiScreen.SETTINGS -> {
            SettingsScreen(
                serverUrl = serverUrl,
                token = token,
                showChangeUrl = showChangeUrl,
                showStatus = showStatus,
                snapshotStatus = snapshotStatus,
                wsStatus = wsStatus,
                pushArmed = pushArmed,
                onBack = { screen.value = UiScreen.WELCOME },
                onApply = {
                    connect()
                },
                onDisconnect = {
                    pushArmed.value = false
                    savePushArmed(false)
                    PushBackend.unregisterCurrentToken(appContext, client)
                    disconnect(jobs, snapshotStatus, wsStatus)
                }
            )
        }
    }
}

@Composable
private fun WelcomeScreen(onContinue: () -> Unit, onHelp: () -> Unit, onSettings: () -> Unit) {
    val logoOffset = animateFloatAsState(
        targetValue = 0f,
        animationSpec = spring(dampingRatio = 0.55f, stiffness = 120f),
        label = "logoOffset"
    ).value

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(
                Brush.radialGradient(
                    colors = listOf(Color(0xFF2A4F7D), Color(0xFF1A3253), Color(0xFF0A0F1E))
                )
            )
            .padding(20.dp)
    ) {
        Column(
            modifier = Modifier.fillMaxSize(),
            verticalArrangement = Arrangement.Center
        ) {
            AnimatedVisibility(
                visible = true,
                enter = fadeIn() + slideInVertically(initialOffsetY = { it / 2 })
            ) {
                FaceClientLogo(modifier = Modifier.size(88.dp).offset(y = logoOffset.dp))
            }
            Spacer(Modifier.height(16.dp))
            Text("FaceClient", color = Color.White, fontSize = 34.sp, fontWeight = FontWeight.ExtraBold)
            Text("Professional AI security for your home", color = Color(0xFFC8DCF6), fontSize = 16.sp)
            Spacer(Modifier.height(22.dp))
            Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                Button(onClick = onContinue) { Text("Open Dashboard") }
                TextButton(onClick = onHelp) { Text("Help") }
            }
            Spacer(Modifier.height(8.dp))
            TextButton(onClick = onSettings) {
                Icon(Icons.Filled.Settings, contentDescription = "Settings")
                Spacer(Modifier.width(6.dp))
                Text("Settings")
            }
        }
    }
}

@Composable
private fun DashboardScreen(
    bitmapState: MutableState<Bitmap?>,
    recentEvents: List<EventItem>,
    timeFmt: SimpleDateFormat,
    onBack: () -> Unit,
    onAllEvents: () -> Unit,
) {
    val showStream = remember { mutableStateOf(false) }
    val showEvents = remember { mutableStateOf(false) }

    LaunchedEffect(Unit) {
        delay(120)
        showStream.value = true
        delay(140)
        showEvents.value = true
    }

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(
                Brush.verticalGradient(
                    colors = listOf(Color(0xFF0A0F1E), Color(0xFF14233C), Color(0xFF1E3558))
                )
            )
    ) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(14.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                IconButton(onClick = onBack) {
                    Icon(Icons.AutoMirrored.Filled.ArrowBack, contentDescription = "Back", tint = Color.White)
                }
                Text("Dashboard", color = Color.White, fontSize = 23.sp, fontWeight = FontWeight.Bold)
                TextButton(onClick = onAllEvents) {
                    Text("All Events")
                }
            }

            AnimatedVisibility(
                visible = showStream.value,
                enter = fadeIn() + slideInVertically(initialOffsetY = { it / 4 })
            ) {
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .weight(1f),
                    contentAlignment = Alignment.Center
                ) {
                    Card(
                        shape = RoundedCornerShape(20.dp),
                        colors = CardDefaults.cardColors(containerColor = Color(0xC00F1A2C)),
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        val bmp = bitmapState.value
                        if (bmp != null) {
                            Image(
                                bitmap = bmp.asImageBitmap(),
                                contentDescription = "Live",
                                modifier = Modifier.fillMaxWidth().aspectRatio(16f / 10f)
                            )
                        } else {
                            Box(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .aspectRatio(16f / 10f)
                                    .padding(16.dp),
                                contentAlignment = Alignment.Center
                            ) {
                                Text("Waiting for live stream...", color = Color(0xFFE1EEFF))
                            }
                        }
                    }
                }
            }

            AnimatedVisibility(
                visible = showEvents.value,
                enter = fadeIn() + slideInVertically(initialOffsetY = { it / 3 })
            ) {
                Card(
                    shape = RoundedCornerShape(18.dp),
                    colors = CardDefaults.cardColors(containerColor = Color(0xB3192740))
                ) {
                    Column(
                        modifier = Modifier.fillMaxWidth().padding(14.dp),
                        verticalArrangement = Arrangement.spacedBy(6.dp)
                    ) {
                        Text("Last Events", color = Color.White, fontSize = 17.sp, fontWeight = FontWeight.Bold)
                        val last5 = recentEvents.take(5)
                        if (last5.isEmpty()) {
                            Text("No events yet.", color = Color(0xFFC8DCF6))
                        } else {
                            last5.forEach { ev ->
                                Text(
                                    "${timeFmt.format(Date(ev.tsMs))}  |  ${ev.name}  |  ${String.format("%.2f", ev.dist)}",
                                    color = Color(0xFFE1EEFF),
                                    fontFamily = FontFamily.Monospace,
                                    fontSize = 12.sp
                                )
                            }
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun SettingsScreen(
    serverUrl: MutableState<String>,
    token: MutableState<String>,
    showChangeUrl: MutableState<Boolean>,
    showStatus: MutableState<Boolean>,
    snapshotStatus: MutableState<String>,
    wsStatus: MutableState<String>,
    pushArmed: MutableState<Boolean>,
    onBack: () -> Unit,
    onApply: () -> Unit,
    onDisconnect: () -> Unit,
) {
    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(
                Brush.verticalGradient(
                    colors = listOf(Color(0xFF111A2D), Color(0xFF1C2F4B), Color(0xFF203C61))
                )
            )
    ) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(14.dp)
                .verticalScroll(rememberScrollState()),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                IconButton(onClick = onBack) {
                    Icon(Icons.AutoMirrored.Filled.ArrowBack, contentDescription = "Back", tint = Color.White)
                }
                Text("Settings", color = Color.White, fontSize = 23.sp, fontWeight = FontWeight.Bold)
                Spacer(Modifier.width(48.dp))
            }

            Card(
                shape = RoundedCornerShape(16.dp),
                colors = CardDefaults.cardColors(containerColor = Color(0xB21A2D4B))
            ) {
                Column(modifier = Modifier.fillMaxWidth().padding(14.dp)) {
                    TextButton(onClick = { showChangeUrl.value = !showChangeUrl.value }) {
                        Text(if (showChangeUrl.value) "Hide Change URL" else "Change URL", color = Color.White)
                    }

                    if (showChangeUrl.value) {
                        Spacer(Modifier.height(8.dp))
                        OutlinedTextField(
                            value = serverUrl.value,
                            onValueChange = { serverUrl.value = it },
                            label = { Text("Server URL") },
                            modifier = Modifier.fillMaxWidth(),
                            singleLine = true
                        )
                        Spacer(Modifier.height(8.dp))
                        OutlinedTextField(
                            value = token.value,
                            onValueChange = { token.value = it },
                            label = { Text("Token (optional)") },
                            modifier = Modifier.fillMaxWidth(),
                            singleLine = true
                        )
                        Spacer(Modifier.height(10.dp))
                        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                            Button(onClick = onApply) { Text("Save + Connect") }
                            Button(onClick = onDisconnect) { Text("Disconnect") }
                        }
                    }
                }
            }

            Card(
                shape = RoundedCornerShape(16.dp),
                colors = CardDefaults.cardColors(containerColor = Color(0xB21A2D4B))
            ) {
                Column(modifier = Modifier.fillMaxWidth().padding(14.dp)) {
                    TextButton(onClick = { showStatus.value = !showStatus.value }) {
                        Text(if (showStatus.value) "Hide Status" else "Status", color = Color.White)
                    }

                    if (showStatus.value) {
                        Spacer(Modifier.height(8.dp))
                        Text("Snapshot: ${snapshotStatus.value}", color = Color(0xFFE1EEFF), fontFamily = FontFamily.Monospace)
                        Text("WebSocket: ${wsStatus.value}", color = Color(0xFFE1EEFF), fontFamily = FontFamily.Monospace)
                        Text("Push: ${if (pushArmed.value) "ARMED" else "DISARMED"}", color = Color(0xFFE1EEFF), fontFamily = FontFamily.Monospace)
                    }
                }
            }
        }
    }
}

@Composable
private fun FaceClientLogo(modifier: Modifier = Modifier) {
    Box(
        modifier = modifier
            .clip(CircleShape)
            .background(
                Brush.linearGradient(
                    colors = listOf(Color(0xFF41C9E2), Color(0xFF2D6CDF), Color(0xFF1C4BA6))
                )
            ),
        contentAlignment = Alignment.Center
    ) {
        Text("FC", color = Color.White, fontWeight = FontWeight.Bold, fontSize = 26.sp)
    }
}

private data class EventItem(
    val tsMs: Long,
    val name: String,
    val dist: Double,
)

private fun isToday(tsMs: Long): Boolean {
    val now = Calendar.getInstance()
    val c = Calendar.getInstance().apply { timeInMillis = tsMs }
    return now.get(Calendar.YEAR) == c.get(Calendar.YEAR) &&
        now.get(Calendar.DAY_OF_YEAR) == c.get(Calendar.DAY_OF_YEAR)
}

private data class Jobs(
    val streamJob: Job,
    val eventPollJob: Job,
    val webSocket: WebSocket,
)

private fun disconnect(
    jobs: MutableState<Jobs?>,
    snapshotStatus: MutableState<String>,
    wsStatus: MutableState<String>
) {
    jobs.value?.let {
        it.streamJob.cancel()
        it.eventPollJob.cancel()
        it.webSocket.close(1000, "bye")
    }
    jobs.value = null
    snapshotStatus.value = "disconnected"
    wsStatus.value = "disconnected"
}

private fun fetchLastEvent(client: OkHttpClient, url: String): JSONObject? {
    val req = Request.Builder().url(url).build()
    client.newCall(req).execute().use { resp ->
        if (!resp.isSuccessful) return null
        val body = resp.body?.string() ?: return null
        if (body.isBlank()) return null
        return JSONObject(body)
    }
}

private fun fetchSnapshot(client: OkHttpClient, url: String): Bitmap? {
    val req = Request.Builder().url(url).build()
    client.newCall(req).execute().use { resp ->
        if (!resp.isSuccessful) return null
        val bytes = resp.body?.bytes() ?: return null
        return BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
    }
}
