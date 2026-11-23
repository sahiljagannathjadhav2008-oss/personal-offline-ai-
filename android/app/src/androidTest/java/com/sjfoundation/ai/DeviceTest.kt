package com.sjfoundation.ai

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.sjfoundation.ai.data.ConversationStore
import com.sjfoundation.ai.data.Msg
import com.sjfoundation.ai.data.SecureStore
import com.sjfoundation.ai.device.DeviceMonitor
import com.sjfoundation.ai.engine.SjEngine
import com.sjfoundation.ai.engine.SjwLoader
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

/** Instrumented tests (need an emulator/device). Run via the manual "instrumented" workflow job. */
@RunWith(AndroidJUnit4::class)
class DeviceTest {
    private val ctx = InstrumentationRegistry.getInstrumentation().targetContext

    @Test fun keystoreEncryptionRoundTripAndCiphertextIsNotPlaintext() {
        val s = SecureStore(); val plain = "मराठी secret".toByteArray()
        val enc = s.encrypt(plain)
        assertFalse(String(enc, Charsets.ISO_8859_1).contains("secret"))
        assertArrayEquals(plain, s.decrypt(enc))
    }

    @Test fun storedConversationFileIsEncryptedAndDeletable() {
        val store = ConversationStore(ctx, SecureStore())
        store.deleteAllLocalData()
        val c = store.newConversation(); c.title = "t"; c.messages.add(Msg("user", "plaintext-marker"))
        store.save(c)
        val f = File(ctx.filesDir, "conv/${c.id}.enc")
        assertTrue(f.exists()); assertFalse(String(f.readBytes(), Charsets.ISO_8859_1).contains("plaintext-marker"))
        assertEquals("plaintext-marker", store.load(c.id)!!.messages[0].text)
        store.deleteAllLocalData()
        assertFalse(f.exists()); assertEquals(0, store.count())
    }

    @Test fun deviceSnapshotAndModelInference() {
        val d = DeviceMonitor(ctx).snapshot()
        assertTrue(d.totalRamMb > 0 && d.cores > 0)
        val bytes = ctx.assets.open("sj_model_int8.sjw").use { it.readBytes() }
        val eng = SjEngine(SjwLoader.load(bytes), 2, 64)
        assertEquals(3480, eng.step(5).size)
    }

    @Test fun appHoldsNoInternetPermission() {
        val info = ctx.packageManager.getPackageInfo(ctx.packageName, android.content.pm.PackageManager.GET_PERMISSIONS)
        assertFalse((info.requestedPermissions ?: emptyArray()).contains("android.permission.INTERNET"))
    }
}
