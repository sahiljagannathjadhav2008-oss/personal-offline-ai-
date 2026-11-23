package com.sjfoundation.ai

import com.sjfoundation.ai.data.InputGuard
import com.sjfoundation.ai.device.AdaptivePolicy
import com.sjfoundation.ai.device.Decision
import com.sjfoundation.ai.device.DeviceInfo
import com.sjfoundation.ai.device.Profile
import com.sjfoundation.ai.engine.ModelFormatException
import com.sjfoundation.ai.engine.Sampler
import com.sjfoundation.ai.engine.SjEngine
import com.sjfoundation.ai.engine.SjTokenizer
import com.sjfoundation.ai.engine.SjwLoader
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File
import kotlin.math.abs

private val assets = File("src/main/assets")
private fun vectors() = JSONObject(File("src/test/resources/test_vectors.json").readText())
private val spark = DeviceInfo(3700, 1500, false, 8, "arm64-v8a", 0, 80, false, false, 20000)
private val poco = DeviceInfo(11800, 6000, false, 8, "arm64-v8a", 0, 80, false, false, 100000)

class TokenizerTest {
    private val tok = SjTokenizer(File(assets, "sj_tokenizer.json").readText())

    @Test fun matchesPythonSentencePieceVectors() {
        val arr = vectors().getJSONArray("tokenizer")
        for (i in 0 until arr.length()) {
            val o = arr.getJSONObject(i)
            val ids = o.getJSONArray("ids")
            val expected = IntArray(ids.length()) { ids.getInt(it) }
            assertEquals(o.getString("text"), expected.toList(), tok.encode(o.getString("text")).toList())
        }
    }

    @Test fun roundTripsMarathiHindiEnglishAndEmoji() {
        for (s in listOf("आज हवामान छान आहे.", "मुझे चाय पसंद है.", "I like tea.", "😀 ₹500")) {
            assertEquals(s, tok.decode(tok.encode(s).toList()))
        }
    }

    @Test fun emptyInputGivesNoTokens() = assertEquals(0, tok.encode("   ").size)
}

class EngineTest {
    private val bytes = File(assets, "sj_model_int8.sjw").readBytes()
    private val model = SjwLoader.load(bytes)

    @Test fun logitsMatchPythonReferenceSingleAndMultiThread() {
        val lg = vectors().getJSONObject("logits")
        val inArr = lg.getJSONArray("input_ids")
        val ids = IntArray(inArr.length()) { inArr.getInt(it) }
        val first8 = lg.getJSONArray("last_logits_first8")
        for (threads in listOf(1, 3)) {
            val eng = SjEngine(model, threads, 128)
            var out = FloatArray(0)
            for (t in ids) out = eng.step(t)
            for (i in 0 until 8) assertEquals(first8.get(i).toString().toDouble(), out[i].toDouble(), lg.getDouble("tolerance"))
            assertEquals(lg.getInt("last_argmax"), Sampler.argmax(out))
            eng.close()
        }
    }

    @Test fun corruptedModelIsRejected() {
        val bad = bytes.copyOf()
        bad[bad.size - 3] = (bad[bad.size - 3] + 1).toByte()
        val r = runCatching { SjwLoader.load(bad) }
        assertTrue(r.exceptionOrNull() is ModelFormatException)
    }

    @Test fun contextLimitIsEnforced() {
        val eng = SjEngine(model, 1, 8)
        repeat(8) { eng.step(5) }
        assertTrue(runCatching { eng.step(5) }.exceptionOrNull() is IllegalStateException)
    }
}

class PolicyTest {
    @Test fun lowPowerProfileForSpark7TClass() {
        val c = AdaptivePolicy.choose(spark, 128)
        assertEquals(Profile.LOW_POWER, c.profile); assertTrue(c.threads <= 2); assertTrue(c.maxContext <= 96)
    }
    @Test fun highPerformanceProfileForPocoClass() {
        val c = AdaptivePolicy.choose(poco, 128)
        assertEquals(Profile.HIGH_PERFORMANCE, c.profile); assertEquals(128, c.maxContext)
    }
    @Test fun thermalMemoryBatteryHandling() {
        val base = AdaptivePolicy.choose(poco, 128)
        assertTrue(AdaptivePolicy.adjust(base, poco.copy(thermalStatus = 3)) is Decision.Stop)
        assertTrue(AdaptivePolicy.adjust(base, poco.copy(lowMemory = true)) is Decision.Stop)
        val mod = AdaptivePolicy.adjust(base, poco.copy(thermalStatus = 2)) as Decision.Run
        assertEquals(1, mod.threads); assertTrue(mod.delayMs > 0)
        assertEquals(1, (AdaptivePolicy.adjust(base, poco.copy(batteryPct = 10)) as Decision.Run).threads)
    }
}

class GuardAndManifestTest {
    @Test fun sanitizeStripsControlAndCaps() {
        val s = InputGuard.sanitize("a\u0000b" + "x".repeat(2000))
        assertFalse(s.contains('\u0000')); assertEquals(InputGuard.MAX_CHARS, s.length)
    }
    @Test fun identityAnswersInThreeLanguages() {
        assertTrue(InputGuard.identityAnswer("Who created SJ Foundation AI?", "en")!!.contains("Sahil Jadhav"))
        assertTrue(InputGuard.identityAnswer("SJ Foundation AI ला कोणी बनवले?", "mr")!!.contains("सहिल जाधव"))
        assertTrue(InputGuard.identityAnswer("SJ Foundation AI को किसने बनाया?", "hi")!!.contains("सहिल जाधव"))
        assertNull(InputGuard.identityAnswer("Who built the pyramids?", "en"))
    }
    @Test fun manifestRequestsNoPermissionsAndDisablesBackup() {
        val m = File("src/main/AndroidManifest.xml").readText()
        assertFalse(m.contains("uses-permission")); assertFalse(m.contains("INTERNET"))
        assertTrue(m.contains("android:allowBackup=\"false\""))
    }
}
