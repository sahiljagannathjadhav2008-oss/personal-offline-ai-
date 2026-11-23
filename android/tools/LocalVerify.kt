// Runs the pure-Kotlin logic (engine, tokenizer, policy, input guard) on a plain JVM against test vectors
// produced by the Python reference. Used for sandbox verification; the same checks exist as JUnit tests.
import com.sjfoundation.ai.data.InputGuard
import com.sjfoundation.ai.device.*
import com.sjfoundation.ai.engine.*
import org.json.JSONObject
import java.io.File
import kotlin.math.abs

var failed = 0
fun check(name: String, ok: Boolean, detail: String = "") {
    println((if (ok) "PASS " else "FAIL ") + name + (if (detail.isNotEmpty()) "  [$detail]" else ""))
    if (!ok) failed++
}

fun main() {
    val assets = File("app/src/main/assets")
    val vec = JSONObject(File("app/src/test/resources/test_vectors.json").readText())
    val tok = SjTokenizer(File(assets, "sj_tokenizer.json").readText())
    val tv = vec.getJSONArray("tokenizer")
    var allTok = true
    for (i in 0 until tv.length()) {
        val o = tv.getJSONObject(i); val ids = o.getJSONArray("ids")
        val exp = IntArray(ids.length()) { ids.getInt(it) }
        if (!tok.encode(o.getString("text")).contentEquals(exp)) { allTok = false; println("  mismatch: " + o.getString("text")) }
    }
    check("tokenizer encode matches Python/SentencePiece (${tv.length()} cases)", allTok)
    check("tokenizer decode round-trip", tok.decode(tok.encode("आज हवामान छान आहे.").toList()) == "आज हवामान छान आहे.")
    check("tokenizer emoji/byte-fallback round-trip", tok.decode(tok.encode("😀 ₹500").toList()) == "😀 ₹500")

    val bytes = File(assets, "sj_model_int8.sjw").readBytes()
    val model = SjwLoader.load(bytes)
    val lg = vec.getJSONObject("logits"); val inArr = lg.getJSONArray("input_ids")
    val ids = IntArray(inArr.length()) { inArr.getInt(it) }
    val first8 = lg.getJSONArray("last_logits_first8"); val tol = lg.getDouble("tolerance")
    for (threads in listOf(1, 3)) {
        val eng = SjEngine(model, threads, 128)
        var out = FloatArray(0)
        for (t in ids) out = eng.step(t)
        var maxd = 0.0
        for (i in 0 until 8) maxd = maxOf(maxd, abs(out[i] - first8.get(i).let { (it as Number).toDouble() }))
        check("engine logits match Python reference (threads=$threads)", maxd < tol && Sampler.argmax(out) == lg.getInt("last_argmax"), "max|diff|=%.5f".format(maxd))
        eng.close()
    }
    val bad = bytes.copyOf(); bad[bad.size - 3] = (bad[bad.size - 3] + 1).toByte()
    check("corrupted model rejected", runCatching { SjwLoader.load(bad) }.exceptionOrNull() is ModelFormatException)
    check("bad magic rejected", runCatching { SjwLoader.load(ByteArray(64)) }.exceptionOrNull() is ModelFormatException)
    val e2 = SjEngine(model, 1, 16)
    repeat(16) { e2.step(5) }
    check("context limit enforced", runCatching { e2.step(5) }.exceptionOrNull() is IllegalStateException)

    // generation speed + sample output (greedy)
    val eng = SjEngine(model, 1, 128)
    val prompt = tok.encode("आज हवामान")
    var l = FloatArray(0); for (t in prompt) l = eng.step(t)
    val gen = ArrayList<Int>(); val t0 = System.nanoTime()
    while (gen.size < 20) { val n = Sampler.argmax(l); if (n == tok.eosId) break; gen.add(n); l = eng.step(n) }
    val dt = (System.nanoTime() - t0) / 1e9
    println("INFO greedy continuation: '" + tok.decode(gen) + "'  (${gen.size} tokens, %.1f tok/s, 1 thread, this sandbox CPU)".format(gen.size / dt))

    // adaptive policy
    val spark = DeviceInfo(3700, 1500, false, 8, "arm64-v8a", 0, 80, false, false, 20000)
    val poco = DeviceInfo(11800, 6000, false, 8, "arm64-v8a", 0, 80, false, false, 100000)
    check("policy: Spark 7T-class -> LOW_POWER, <=2 threads, ctx<=96", AdaptivePolicy.choose(spark, 128).let { it.profile == Profile.LOW_POWER && it.threads <= 2 && it.maxContext <= 96 })
    check("policy: POCO X7 Pro-class -> HIGH_PERFORMANCE, 6 threads, ctx 128", AdaptivePolicy.choose(poco, 128).let { it.profile == Profile.HIGH_PERFORMANCE && it.threads == 6 && it.maxContext == 128 })
    check("policy: override respected", AdaptivePolicy.choose(poco, 128, Profile.LOW_POWER).profile == Profile.LOW_POWER)
    val base = AdaptivePolicy.choose(poco, 128)
    check("policy: severe thermal stops", AdaptivePolicy.adjust(base, poco.copy(thermalStatus = 3)) is Decision.Stop)
    check("policy: low memory stops", AdaptivePolicy.adjust(base, poco.copy(lowMemory = true)) is Decision.Stop)
    check("policy: <150MB free stops", AdaptivePolicy.adjust(base, poco.copy(availRamMb = 100)) is Decision.Stop)
    check("policy: moderate thermal -> 1 thread + delay", AdaptivePolicy.adjust(base, poco.copy(thermalStatus = 2)).let { it is Decision.Run && it.threads == 1 && it.delayMs > 0 })
    check("policy: battery saver caps threads at 2", AdaptivePolicy.adjust(base, poco.copy(powerSave = true)).let { it is Decision.Run && it.threads == 2 })
    check("policy: low battery -> 1 thread", AdaptivePolicy.adjust(base, poco.copy(batteryPct = 10)).let { it is Decision.Run && it.threads == 1 })
    check("policy: normal conditions keep config", AdaptivePolicy.adjust(base, poco).let { it is Decision.Run && it.threads == base.threads && it.delayMs == 0L })

    // input guard / identity
    check("guard: control chars stripped, length capped", InputGuard.sanitize("a\u0000b\u0007c" + "x".repeat(2000)).let { !it.contains('\u0000') && it.length == InputGuard.MAX_CHARS })
    check("identity en", InputGuard.identityAnswer("Who created SJ Foundation AI?", "en")?.contains("Sahil Jadhav") == true)
    check("identity mr", InputGuard.identityAnswer("SJ Foundation AI ला कोणी बनवले?", "mr")?.contains("सहिल जाधव") == true)
    check("identity hi", InputGuard.identityAnswer("SJ Foundation AI को किसने बनाया?", "hi")?.contains("सहिल जाधव") == true)
    check("identity: unrelated question not hijacked", InputGuard.identityAnswer("What is the weather today?", "en") == null && InputGuard.identityAnswer("Who built the pyramids?", "en") == null)

    println(if (failed == 0) "ALL KOTLIN LOCAL CHECKS PASSED" else "$failed CHECK(S) FAILED")
    kotlin.system.exitProcess(if (failed == 0) 0 else 1)
}
