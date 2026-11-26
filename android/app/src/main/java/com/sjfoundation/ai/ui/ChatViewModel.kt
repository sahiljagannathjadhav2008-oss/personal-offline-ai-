package com.sjfoundation.ai.ui

import android.app.Application
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.sjfoundation.ai.data.Conversation
import com.sjfoundation.ai.data.ConversationStore
import com.sjfoundation.ai.data.InputGuard
import com.sjfoundation.ai.data.Msg
import com.sjfoundation.ai.data.SecureStore
import com.sjfoundation.ai.data.Settings
import com.sjfoundation.ai.device.AdaptivePolicy
import com.sjfoundation.ai.device.DeviceInfo
import com.sjfoundation.ai.device.DeviceMonitor
import com.sjfoundation.ai.device.Decision
import com.sjfoundation.ai.device.Profile
import com.sjfoundation.ai.device.RuntimeConfig
import com.sjfoundation.ai.engine.Sampler
import com.sjfoundation.ai.engine.SjEngine
import com.sjfoundation.ai.engine.SjTokenizer
import com.sjfoundation.ai.engine.SjwLoader
import com.sjfoundation.ai.engine.SjwModel
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Job
import kotlinx.coroutines.NonCancellable
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONObject
import kotlin.random.Random

class ChatViewModel(app: Application) : AndroidViewModel(app) {
    val settings = Settings(app)
    private val secure = SecureStore()
    val store = ConversationStore(app, secure)
    private val monitor = DeviceMonitor(app)

    var modelStatus by mutableStateOf("Loading model…")
    var modelReady by mutableStateOf(false)
    var generating by mutableStateOf(false)
    var statusLine by mutableStateOf("")
    var error by mutableStateOf<String?>(null)
    var device by mutableStateOf<DeviceInfo?>(null)
    var runtime by mutableStateOf<RuntimeConfig?>(null)
    var lastTokPerSec by mutableStateOf(0.0)
    var modelCard by mutableStateOf<JSONObject?>(null)
    var language by mutableStateOf(settings.language)
    var profileOverride by mutableStateOf(settings.profileOverride)
    var temperature by mutableStateOf(settings.temperature)
    var history = mutableStateListOf<Conversation>()
    var current by mutableStateOf(store.newConversation())
    val messages = mutableStateListOf<Msg>()

    private var model: SjwModel? = null
    private var tokenizer: SjTokenizer? = null
    private var engine: SjEngine? = null
    private var job: Job? = null

    init { loadModel(); refreshHistory() }

    fun refreshDevice() { device = monitor.snapshot() }

    private fun chosenConfig(): RuntimeConfig? {
        val m = model ?: return null
        val d = monitor.snapshot().also { device = it }
        val ov = runCatching { Profile.valueOf(profileOverride) }.getOrNull()
        return AdaptivePolicy.choose(d, m.config.maxSeq, ov).also { runtime = it }
    }

    fun loadModel() {
        viewModelScope.launch(Dispatchers.Default) {
            try {
                val ctx = getApplication<Application>()
                val bytes = ctx.assets.open("sj_model_int8.sjw").use { it.readBytes() }
                val tok = SjTokenizer(ctx.assets.open("sj_tokenizer.json").use { String(it.readBytes(), Charsets.UTF_8) })
                val card = JSONObject(ctx.assets.open("model_card.json").use { String(it.readBytes(), Charsets.UTF_8) })
                val m = SjwLoader.load(bytes)   // verifies SHA-256 integrity
                withContext(Dispatchers.Main) {
                    model = m; tokenizer = tok; modelCard = card
                    val cfg = chosenConfig()
                    if (cfg != null) engine = SjEngine(m, cfg.threads, cfg.maxContext)
                    modelReady = true; modelStatus = "Model ready (CPU, INT8)"
                }
            } catch (e: Throwable) {
                withContext(Dispatchers.Main) { modelStatus = "Model failed to load"; error = "Could not load the model: ${e.message}" }
            }
        }
    }

    /** Re-applies the processing profile (after the user changes it). */
    fun reconfigure() {
        val m = model ?: return
        val cfg = chosenConfig() ?: return
        engine?.close(); engine = SjEngine(m, cfg.threads, cfg.maxContext)
    }

    fun changeLanguage(l: String) { language = l; settings.language = l }
    fun setProfile(p: String) { profileOverride = p; settings.profileOverride = p; reconfigure() }
    fun setTemp(t: Float) { temperature = t; settings.temperature = t }

    fun send(raw: String) {
        if (generating || !modelReady) return
        val text = InputGuard.sanitize(raw)
        if (text.isEmpty()) return
        messages.add(Msg("user", text))
        if (current.title.isEmpty()) current.title = text.take(40)
        val identity = InputGuard.identityAnswer(text, language)
        if (identity != null) { messages.add(Msg("ai", identity)); persist(); return }
        val tok = tokenizer ?: return; val cfg = runtime ?: chosenConfig() ?: return
        val eng = engine ?: return
        generating = true; error = null
        messages.add(Msg("ai", ""))
        val idx = messages.lastIndex
        job = viewModelScope.launch(Dispatchers.Default) {
            try {
                eng.reset()
                var prompt = tok.encode(text).toList()
                val room = eng.maxContext - 8
                if (prompt.size > room) prompt = prompt.takeLast(room)
                val rnd = Random(System.nanoTime())
                var logits = FloatArray(0)
                for (t in prompt) logits = eng.step(t)
                val out = ArrayList<Int>()
                val t0 = System.nanoTime()
                val limit = minOf(cfg.maxNewTokens, eng.maxContext - prompt.size)
                val banned = setOf(0, 1, 2)
                var stopReason: String? = null
                while (out.size < limit && isActive) {
                    val d = monitor.snapshot()
                    withContext(Dispatchers.Main) { device = d }
                    when (val dec = AdaptivePolicy.adjust(cfg, d)) {
                        is Decision.Stop -> { stopReason = dec.reason; break }
                        is Decision.Run -> { if (dec.threads != eng.threads) eng.setThreads(dec.threads); if (dec.delayMs > 0) delay(dec.delayMs) }
                    }
                    val next = Sampler.sample(logits, temperature, 20, rnd, banned)
                    if (next == tok.eosId) break
                    out.add(next)
                    val s = tok.decode(out)
                    withContext(Dispatchers.Main) { if (idx in messages.indices) messages[idx] = Msg("ai", s); statusLine = "${out.size} tokens" }
                    if (out.size >= limit) break
                    logits = eng.step(next)
                }
                val secs = (System.nanoTime() - t0) / 1e9
                withContext(Dispatchers.Main) {
                    if (out.isNotEmpty() && secs > 0) lastTokPerSec = out.size / secs
                    if (stopReason != null) error = stopReason
                    if (idx in messages.indices && messages[idx].text.isEmpty()) messages[idx] = Msg("ai", "(no output)")
                    statusLine = ""
                }
            } catch (e: CancellationException) {
                // user pressed Stop: keep whatever text was generated so far
            } catch (e: Throwable) {
                withContext(NonCancellable + Dispatchers.Main) { error = "Generation failed: ${e.message}"; if (idx in messages.indices && messages[idx].text.isEmpty()) messages[idx] = Msg("ai", "(error)") }
            } finally {
                withContext(NonCancellable + Dispatchers.Main) {
                    if (idx in messages.indices && messages[idx].text.isEmpty()) messages[idx] = Msg("ai", "(stopped)")
                    statusLine = ""; generating = false; persist()
                }
            }
        }
    }

    fun stop() { job?.cancel() }

    private fun persist() {
        current.messages.clear(); current.messages.addAll(messages)
        current.updated = System.currentTimeMillis()
        viewModelScope.launch(Dispatchers.IO) { runCatching { store.save(current) }; withContext(Dispatchers.Main) { refreshHistory() } }
    }

    fun newConversation() { job?.cancel(); generating = false; current = store.newConversation(); messages.clear(); error = null }
    fun clearConversation() { store.delete(current.id); newConversation(); refreshHistory() }
    fun open(c: Conversation) { job?.cancel(); generating = false; current = c; messages.clear(); messages.addAll(c.messages) }
    fun deleteConversation(c: Conversation) { store.delete(c.id); if (c.id == current.id) newConversation(); refreshHistory() }

    fun refreshHistory() {
        viewModelScope.launch(Dispatchers.IO) {
            val l = runCatching { store.list() }.getOrDefault(emptyList())
            withContext(Dispatchers.Main) { history.clear(); history.addAll(l) }
        }
    }

    fun deleteAllData() {
        job?.cancel(); store.deleteAllLocalData(); newConversation(); history.clear()
        language = "en"; profileOverride = "auto"; temperature = 0.7f
    }

    override fun onCleared() { engine?.close() }
}