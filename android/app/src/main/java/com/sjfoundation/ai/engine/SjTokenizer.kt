package com.sjfoundation.ai.engine

import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.text.Normalizer
import kotlin.math.exp
import kotlin.random.Random

/** SentencePiece-BPE compatible tokenizer (mirrors sjai/export.py:BPETokenizer). */
class SjTokenizer(json: String) {
    private val type = ArrayList<Char>()
    private val piece = ArrayList<String>()
    private val id = HashMap<String, Int>()
    private val score = HashMap<String, Float>()
    private val byteId = HashMap<Int, Int>()
    val unkId: Int
    val eosId: Int

    init {
        val arr = JSONObject(json).getJSONArray("pieces")
        var unk = 1
        var eos = 3
        for (i in 0 until arr.length()) {
            val o = arr.getJSONObject(i)
            val p = o.getString("p")
            val t = o.getString("t")[0]
            type.add(t)
            piece.add(p)
            when (t) {
                'n' -> { id[p] = i; score[p] = o.getDouble("s").toFloat() }
                'b' -> byteId[p.substring(3, 5).toInt(16)] = i
                'u' -> unk = i
            }
            if (p == "</s>") eos = i
        }
        unkId = unk
        eosId = eos
    }

    val vocabSize: Int get() = piece.size

    fun encode(text: String): IntArray {
        val norm = Normalizer.normalize(text, Normalizer.Form.NFKC)
            .split(Regex("\\s+")).filter { it.isNotEmpty() }.joinToString(" ")
        if (norm.isEmpty()) return IntArray(0)
        val full = "\u2581" + norm.replace(' ', '\u2581')
        val syms = ArrayList<String>()
        var i = 0
        while (i < full.length) {
            val n = Character.charCount(full.codePointAt(i))
            syms.add(full.substring(i, i + n))
            i += n
        }
        while (true) {
            var best = Float.NEGATIVE_INFINITY
            var bi = -1
            for (k in 0 until syms.size - 1) {
                val s = score[syms[k] + syms[k + 1]] ?: continue
                if (s > best) { best = s; bi = k }
            }
            if (bi < 0) break
            syms[bi] = syms[bi] + syms[bi + 1]
            syms.removeAt(bi + 1)
        }
        val out = ArrayList<Int>()
        for (s in syms) {
            val tid = id[s]
            if (tid != null) out.add(tid)
            else for (b in s.toByteArray(Charsets.UTF_8)) out.add(byteId[b.toInt() and 0xFF] ?: unkId)
        }
        return out.toIntArray()
    }

    fun decode(ids: List<Int>): String {
        val buf = ByteArrayOutputStream()
        for (t in ids) {
            if (t !in piece.indices) continue
            when (type[t]) {
                'b' -> buf.write(piece[t].substring(3, 5).toInt(16))
                'n' -> buf.write(piece[t].toByteArray(Charsets.UTF_8))
                else -> {}
            }
        }
        val s = String(buf.toByteArray(), Charsets.UTF_8).replace('\u2581', ' ')
        return if (s.startsWith(" ")) s.substring(1) else s
    }
}

object Sampler {
    fun argmax(l: FloatArray): Int {
        var b = 0
        for (i in l.indices) if (l[i] > l[b]) b = i
        return b
    }

    fun sample(logits: FloatArray, temperature: Float, topK: Int, rnd: Random, banned: Set<Int> = emptySet()): Int {
        val l = logits.copyOf()
        for (b in banned) if (b in l.indices) l[b] = -Float.MAX_VALUE
        if (temperature <= 1e-3f) return argmax(l)
        val idx = l.indices.sortedByDescending { l[it] }.take(topK.coerceIn(1, l.size))
        val mx = l[idx[0]]
        val w = idx.map { exp(((l[it] - mx) / temperature).toDouble()) }
        var r = rnd.nextDouble() * w.sum()
        for ((i, wi) in w.withIndex()) {
            r -= wi
            if (r <= 0) return idx[i]
        }
        return idx.last()
    }
}
