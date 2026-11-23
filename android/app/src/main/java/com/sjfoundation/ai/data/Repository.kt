package com.sjfoundation.ai.data

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.util.UUID

data class Msg(val role: String, val text: String)
data class Conversation(val id: String, var title: String, var updated: Long, val messages: MutableList<Msg>)

/** Conversations are stored locally, one AES-GCM-encrypted file each. Nothing leaves the device. */
class ConversationStore(private val ctx: Context, private val secure: SecureStore) {
    private val dir get() = File(ctx.filesDir, "conv").apply { mkdirs() }

    fun newConversation() = Conversation(UUID.randomUUID().toString(), "", System.currentTimeMillis(), mutableListOf())

    fun save(c: Conversation) {
        if (c.messages.isEmpty()) return
        val o = JSONObject().put("id", c.id).put("title", c.title).put("updated", c.updated)
        val arr = JSONArray()
        c.messages.forEach { arr.put(JSONObject().put("r", it.role).put("t", it.text)) }
        o.put("m", arr)
        val tmp = File(dir, "${c.id}.tmp")
        tmp.writeBytes(secure.encrypt(o.toString().toByteArray(Charsets.UTF_8)))
        tmp.renameTo(File(dir, "${c.id}.enc"))
    }

    fun load(id: String): Conversation? = runCatching {
        val o = JSONObject(String(secure.decrypt(File(dir, "$id.enc").readBytes()), Charsets.UTF_8))
        val arr = o.getJSONArray("m")
        Conversation(o.getString("id"), o.getString("title"), o.getLong("updated"),
            MutableList(arr.length()) { Msg(arr.getJSONObject(it).getString("r"), arr.getJSONObject(it).getString("t")) })
    }.getOrNull()

    fun list(): List<Conversation> = (dir.listFiles { f -> f.name.endsWith(".enc") } ?: emptyArray())
        .mapNotNull { load(it.name.removeSuffix(".enc")) }.sortedByDescending { it.updated }

    fun delete(id: String) { File(dir, "$id.enc").delete() }
    fun count() = dir.listFiles { f -> f.name.endsWith(".enc") }?.size ?: 0
    fun sizeBytes() = dir.listFiles()?.sumOf { it.length() } ?: 0L

    /** Removes every conversation, all settings and the encryption key. */
    fun deleteAllLocalData() {
        dir.deleteRecursively()
        ctx.getSharedPreferences("sj_settings", Context.MODE_PRIVATE).edit().clear().apply()
        secure.deleteKey()
    }
}

class Settings(ctx: Context) {
    private val p = ctx.getSharedPreferences("sj_settings", Context.MODE_PRIVATE)
    var language: String
        get() = p.getString("lang", "en") ?: "en"
        set(v) = p.edit().putString("lang", v).apply()
    var profileOverride: String   // "auto" | LOW_POWER | BALANCED | HIGH_PERFORMANCE
        get() = p.getString("profile", "auto") ?: "auto"
        set(v) = p.edit().putString("profile", v).apply()
    var temperature: Float
        get() = p.getFloat("temp", 0.7f)
        set(v) = p.edit().putFloat("temp", v).apply()
}
