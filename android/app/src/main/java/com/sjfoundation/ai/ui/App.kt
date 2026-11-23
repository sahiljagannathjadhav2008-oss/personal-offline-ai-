package com.sjfoundation.ai.ui

import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.DrawerValue
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.ModalDrawerSheet
import androidx.compose.material3.ModalNavigationDrawer
import androidx.compose.material3.NavigationDrawerItem
import androidx.compose.material3.NavigationDrawerItemDefaults
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Slider
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.rememberDrawerState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalClipboardManager
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.sjfoundation.ai.data.Conversation
import com.sjfoundation.ai.device.Profile
import kotlinx.coroutines.launch
import java.text.DateFormat
import java.util.Date

val Cyan = Color(0xFF22D3EE)
val Blue = Color(0xFF3B82F6)
val Bg = Color(0xFF0A0A0C)
val Graphite = Color(0xFF16181D)
val Graphite2 = Color(0xFF20232A)

private val scheme = darkColorScheme(
    primary = Cyan, secondary = Blue, background = Bg, surface = Graphite,
    onPrimary = Color.Black, onBackground = Color(0xFFE6EDF3), onSurface = Color(0xFFE6EDF3),
)

enum class Screen(val key: String) {
    Chat("chat"), History("history"), Model("model"), Device("device"), Processing("processing"),
    Language("language"), Privacy("privacy"), Storage("storage"), About("about"),
}

/** UI strings in English, Marathi and Hindi (labels only; the model itself is a separate multilingual LM). */
object Str {
    private val m: Map<String, Array<String>> = mapOf(
        "chat" to arrayOf("Chat", "चॅट", "चैट"),
        "new" to arrayOf("New conversation", "नवीन संभाषण", "नई बातचीत"),
        "history" to arrayOf("Conversation history", "संभाषण इतिहास", "बातचीत का इतिहास"),
        "model" to arrayOf("Model information", "मॉडेल माहिती", "मॉडल जानकारी"),
        "device" to arrayOf("Device performance", "डिव्हाइस कामगिरी", "डिवाइस प्रदर्शन"),
        "processing" to arrayOf("Processing settings", "प्रोसेसिंग सेटिंग्ज", "प्रोसेसिंग सेटिंग्स"),
        "language" to arrayOf("Language settings", "भाषा सेटिंग्ज", "भाषा सेटिंग्स"),
        "privacy" to arrayOf("Privacy and security", "गोपनीयता आणि सुरक्षा", "गोपनीयता और सुरक्षा"),
        "storage" to arrayOf("Storage management", "स्टोरेज व्यवस्थापन", "स्टोरेज प्रबंधन"),
        "about" to arrayOf("About developer", "डेव्हलपरबद्दल", "डेवलपर के बारे में"),
        "send" to arrayOf("Send", "पाठवा", "भेजें"),
        "stop" to arrayOf("Stop", "थांबवा", "रोकें"),
        "copy" to arrayOf("Copy", "कॉपी", "कॉपी"),
        "clear" to arrayOf("Clear conversation", "संभाषण पुसा", "बातचीत हटाएँ"),
        "hint" to arrayOf("Type a message…", "संदेश लिहा…", "संदेश लिखें…"),
        "delete" to arrayOf("Delete", "हटवा", "हटाएँ"),
        "delall" to arrayOf("Delete all local data", "सर्व स्थानिक डेटा हटवा", "सारा स्थानीय डेटा हटाएँ"),
        "delconv" to arrayOf("Delete all conversations", "सर्व संभाषणे हटवा", "सारी बातचीत हटाएँ"),
    )
    fun t(k: String, lang: String): String {
        val a = m[k] ?: return k
        return a[when (lang) { "mr" -> 1; "hi" -> 2; else -> 0 }]
    }
}

@Composable
fun SjTheme(content: @Composable () -> Unit) = MaterialTheme(colorScheme = scheme, content = content)

@Composable
fun CyanButton(text: String, enabled: Boolean = true, onClick: () -> Unit) {
    OutlinedButton(
        onClick = onClick, enabled = enabled, border = BorderStroke(1.5.dp, if (enabled) Cyan else Color.Gray),
        shape = RoundedCornerShape(12.dp), colors = ButtonDefaults.outlinedButtonColors(contentColor = Cyan),
    ) { Text(text) }
}

@Composable
fun Card(modifier: Modifier = Modifier, content: @Composable () -> Unit) {
    Column(
        modifier.fillMaxWidth().background(Graphite, RoundedCornerShape(14.dp))
            .border(1.dp, Graphite2, RoundedCornerShape(14.dp)).padding(14.dp),
    ) { content() }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SjApp(vm: ChatViewModel) {
    var screen by remember { mutableStateOf(Screen.Chat) }
    val drawer = rememberDrawerState(DrawerValue.Closed)
    val scope = rememberCoroutineScope()
    val lang = vm.language
    ModalNavigationDrawer(
        drawerState = drawer,
        drawerContent = {
            ModalDrawerSheet(drawerContainerColor = Graphite) {
                Text("SJ Foundation AI", Modifier.padding(20.dp), color = Cyan, fontSize = 20.sp, fontWeight = FontWeight.Bold)
                NavigationDrawerItem(
                    label = { Text(Str.t("new", lang)) }, selected = false,
                    onClick = { vm.newConversation(); screen = Screen.Chat; scope.launch { drawer.close() } },
                    modifier = Modifier.padding(NavigationDrawerItemDefaults.ItemPadding),
                )
                Screen.values().forEach { s ->
                    NavigationDrawerItem(
                        label = { Text(Str.t(s.key, lang)) }, selected = s == screen,
                        onClick = { screen = s; scope.launch { drawer.close() } },
                        modifier = Modifier.padding(NavigationDrawerItemDefaults.ItemPadding),
                    )
                }
            }
        },
    ) {
        Scaffold(
            containerColor = Bg,
            topBar = {
                TopAppBar(
                    title = { Text(Str.t(screen.key, lang), color = Color.White) },
                    navigationIcon = { CyanButton("☰") { scope.launch { drawer.open() } } },
                    colors = TopAppBarDefaults.topAppBarColors(containerColor = Bg),
                )
            },
        ) { pad ->
            Box(Modifier.padding(pad).fillMaxSize()) {
                when (screen) {
                    Screen.Chat -> ChatScreen(vm)
                    Screen.History -> HistoryScreen(vm) { screen = Screen.Chat }
                    Screen.Model -> ModelScreen(vm)
                    Screen.Device -> DeviceScreen(vm)
                    Screen.Processing -> ProcessingScreen(vm)
                    Screen.Language -> LanguageScreen(vm)
                    Screen.Privacy -> PrivacyScreen(vm)
                    Screen.Storage -> StorageScreen(vm)
                    Screen.About -> AboutScreen()
                }
            }
        }
    }
}

@Composable
fun ChatScreen(vm: ChatViewModel) {
    var input by remember { mutableStateOf("") }
    val lang = vm.language
    val list = rememberLazyListState()
    val clip = LocalClipboardManager.current
    LaunchedEffect(vm.messages.size, vm.messages.lastOrNull()?.text) {
        if (vm.messages.isNotEmpty()) list.animateScrollToItem(vm.messages.lastIndex)
    }
    Column(Modifier.fillMaxSize().imePadding().padding(12.dp)) {
        Text(vm.modelStatus + if (vm.statusLine.isNotEmpty()) " · ${vm.statusLine}" else "", color = Cyan, fontSize = 12.sp)
        vm.error?.let { Text(it, color = Color(0xFFFF6B6B), fontSize = 13.sp, modifier = Modifier.padding(vertical = 4.dp)) }
        LazyColumn(Modifier.weight(1f).fillMaxWidth(), state = list, verticalArrangement = Arrangement.spacedBy(8.dp)) {
            items(vm.messages) { m ->
                val user = m.role == "user"
                Row(Modifier.fillMaxWidth(), horizontalArrangement = if (user) Arrangement.End else Arrangement.Start) {
                    Column(
                        Modifier.widthIn(max = 300.dp)
                            .background(if (user) Color(0xFF0E3A4A) else Graphite, RoundedCornerShape(14.dp))
                            .border(1.dp, if (user) Cyan else Graphite2, RoundedCornerShape(14.dp)).padding(10.dp),
                    ) {
                        Text(m.text.ifEmpty { "…" }, color = Color.White)
                        if (!user && m.text.isNotEmpty()) {
                            Text(
                                Str.t("copy", lang), color = Cyan, fontSize = 12.sp,
                                modifier = Modifier.padding(top = 6.dp).clickable { clip.setText(AnnotatedString(m.text)) },
                            )
                        }
                    }
                }
            }
        }
        Row(Modifier.fillMaxWidth().padding(top = 8.dp), verticalAlignment = Alignment.CenterVertically) {
            OutlinedTextField(
                value = input, onValueChange = { input = it.take(1000) }, modifier = Modifier.weight(1f),
                placeholder = { Text(Str.t("hint", lang)) }, maxLines = 4,
            )
            Box(Modifier.padding(start = 8.dp)) {
                if (vm.generating) CyanButton(Str.t("stop", lang)) { vm.stop() }
                else CyanButton(Str.t("send", lang), enabled = vm.modelReady && input.isNotBlank()) { vm.send(input); input = "" }
            }
        }
        if (vm.messages.isNotEmpty()) {
            Box(Modifier.padding(top = 6.dp)) { CyanButton(Str.t("clear", lang)) { vm.clearConversation() } }
        }
    }
}

@Composable
fun HistoryScreen(vm: ChatViewModel, onOpen: () -> Unit) {
    val lang = vm.language
    val df = DateFormat.getDateTimeInstance(DateFormat.MEDIUM, DateFormat.SHORT)
    if (vm.history.isEmpty()) {
        Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) { Text("No saved conversations yet.", color = Color.Gray) }
        return
    }
    LazyColumn(Modifier.fillMaxSize().padding(12.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
        items(vm.history, key = { it.id }) { c: Conversation ->
            Card {
                Text(c.title.ifEmpty { "(untitled)" }, color = Color.White, fontWeight = FontWeight.Bold)
                Text(df.format(Date(c.updated)) + " · ${c.messages.size} messages", color = Color.Gray, fontSize = 12.sp)
                Row(Modifier.padding(top = 8.dp), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    CyanButton("Open") { vm.open(c); onOpen() }
                    CyanButton(Str.t("delete", lang)) { vm.deleteConversation(c) }
                }
            }
        }
    }
}

@Composable
fun Page(content: @Composable () -> Unit) {
    Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(12.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) { content() }
}

@Composable
fun Row2(k: String, v: String) {
    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
        Text(k, color = Color.Gray); Text(v, color = Color.White, modifier = Modifier.widthIn(max = 200.dp))
    }
}

@Composable
fun ModelScreen(vm: ChatViewModel) {
    val c = vm.modelCard
    Page {
        Card {
            if (c == null) Text("Model not loaded.", color = Color.White) else {
                Row2("Name", c.optString("name"))
                Row2("Parameters", "%,d".format(c.optLong("parameters")))
                Row2("Architecture", c.optString("architecture"))
                Row2("Vocabulary", c.optInt("vocab_size").toString())
                Row2("Context length", c.optInt("max_seq").toString())
                Row2("Deployed weights", c.optString("quantization"))
                Row2("Training steps (best)", c.optInt("best_step").toString())
                Row2("Validation perplexity", c.optString("val_perplexity"))
                Row2("Dataset version", c.optString("dataset_version"))
                Row2("Training data", c.optString("data_license"))
                Row2("Compute", "CPU only")
            }
        }
        Card {
            Text("What this model is", color = Cyan, fontWeight = FontWeight.Bold)
            Text(c?.optString("limitations") ?: "", color = Color.White, fontSize = 13.sp)
        }
    }
}

@Composable
fun DeviceScreen(vm: ChatViewModel) {
    LaunchedEffect(Unit) { vm.refreshDevice() }
    val d = vm.device
    val rt = vm.runtime
    Page {
        Card {
            if (d == null) Text("Reading device…", color = Color.White) else {
                Row2("Total RAM", "${d.totalRamMb} MB")
                Row2("Available RAM", "${d.availRamMb} MB")
                Row2("CPU cores", d.cores.toString())
                Row2("ABI", d.abi)
                Row2("Thermal status", d.thermalStatus.toString())
                Row2("Battery", if (d.batteryPct >= 0) "${d.batteryPct}%" + if (d.charging) " (charging)" else "" else "unknown")
                Row2("Battery saver", if (d.powerSave) "on" else "off")
                Row2("Free storage", "${d.freeStorageMb} MB")
                Row2("GPU/NPU acceleration", "not used (CPU only)")
            }
        }
        Card {
            Text("Active inference configuration", color = Cyan, fontWeight = FontWeight.Bold)
            if (rt != null) {
                Row2("Profile", rt.profile.name)
                Row2("CPU threads", rt.threads.toString())
                Row2("Context window", "${rt.maxContext} tokens")
                Row2("Max new tokens", rt.maxNewTokens.toString())
                Row2("Last speed", "%.1f tokens/s".format(vm.lastTokPerSec))
            }
        }
        CyanButton("Refresh") { vm.refreshDevice() }
    }
}

@Composable
fun ProcessingScreen(vm: ChatViewModel) {
    Page {
        Card {
            Text("Performance profile", color = Cyan, fontWeight = FontWeight.Bold)
            Text("Auto chooses from your device's RAM and cores. The app always slows down or stops if the phone gets hot or low on memory.", color = Color.Gray, fontSize = 12.sp)
            listOf("auto", Profile.LOW_POWER.name, Profile.BALANCED.name, Profile.HIGH_PERFORMANCE.name).forEach { p ->
                Row(Modifier.fillMaxWidth().clickable { vm.setProfile(p) }.padding(vertical = 8.dp), verticalAlignment = Alignment.CenterVertically) {
                    Text(if (vm.profileOverride == p) "●  " else "○  ", color = Cyan)
                    Text(p, color = Color.White)
                }
            }
        }
        Card {
            Text("Creativity (temperature): %.2f".format(vm.temperature), color = Cyan, fontWeight = FontWeight.Bold)
            Slider(value = vm.temperature, onValueChange = { vm.setTemp(it) }, valueRange = 0f..1.2f)
        }
    }
}

@Composable
fun LanguageScreen(vm: ChatViewModel) {
    Page {
        Card {
            Text("App language (menus and labels)", color = Cyan, fontWeight = FontWeight.Bold)
            listOf("en" to "English", "mr" to "मराठी", "hi" to "हिन्दी").forEach { (code, name) ->
                Row(Modifier.fillMaxWidth().clickable { vm.setLanguage(code) }.padding(vertical = 10.dp)) {
                    Text(if (vm.language == code) "●  " else "○  ", color = Cyan); Text(name, color = Color.White)
                }
            }
            Text("The model reads and writes Marathi, Hindi and English text regardless of this setting; this only changes the interface.", color = Color.Gray, fontSize = 12.sp)
        }
    }
}

@Composable
fun PrivacyScreen(vm: ChatViewModel) {
    var confirm by remember { mutableStateOf(false) }
    Page {
        Card {
            Text("What is stored", color = Cyan, fontWeight = FontWeight.Bold)
            Text(
                "• Your conversations, saved on this device only, encrypted with AES-256-GCM using a key held in the Android Keystore.\n" +
                    "• Your settings (language, performance profile, temperature).\n" +
                    "• Nothing else. No accounts, no analytics, no telemetry.",
                color = Color.White, fontSize = 13.sp,
            )
        }
        Card {
            Text("Network and training", color = Cyan, fontWeight = FontWeight.Bold)
            Text(
                "• This app does not request the INTERNET permission, so it cannot send data off your device.\n" +
                    "• Your conversations are never used to train any model.\n" +
                    "• App backup is disabled so conversations are not copied to cloud backups.\n" +
                    "• The model file's SHA-256 is verified every time it loads.\n" +
                    "• There is no tool or command execution; model output is only displayed as text.",
                color = Color.White, fontSize = 13.sp,
            )
        }
        if (!confirm) CyanButton(Str.t("delall", vm.language)) { confirm = true }
        else {
            Text("This permanently deletes all conversations and settings and the encryption key.", color = Color(0xFFFF6B6B))
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                CyanButton("Yes, delete") { vm.deleteAllData(); confirm = false }
                CyanButton("Cancel") { confirm = false }
            }
        }
    }
}

@Composable
fun StorageScreen(vm: ChatViewModel) {
    val lang = vm.language
    val size = remember(vm.history.size) { vm.store.sizeBytes() }
    Page {
        Card {
            Row2("Saved conversations", vm.history.size.toString())
            Row2("Conversation data size", "%.1f KB".format(size / 1024.0))
            Row2("Model file (in app)", "~9.9 MB")
            Row2("Free device storage", "${vm.device?.freeStorageMb ?: "?"} MB")
        }
        CyanButton(Str.t("delconv", lang)) { vm.history.toList().forEach { vm.store.delete(it.id) }; vm.newConversation(); vm.refreshHistory() }
        CyanButton(Str.t("delall", lang)) { vm.deleteAllData() }
    }
}

@Composable
fun AboutScreen() {
    Page {
        Card {
            Text("SJ Foundation AI", color = Cyan, fontSize = 22.sp, fontWeight = FontWeight.Bold)
            Text("An independent small language model trained from random initialization, running fully on your device.", color = Color.White)
        }
        Card {
            Row2("Developer", "Sahil Jadhav")
            Row2("Version", "0.1.0")
        }
    }
}
