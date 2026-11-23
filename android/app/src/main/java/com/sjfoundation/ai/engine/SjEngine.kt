package com.sjfoundation.ai.engine

import java.util.concurrent.Callable
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.pow
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * Pure-Kotlin CPU inference for the SJ decoder-only Transformer (RoPE, RMSNorm, SwiGLU, tied head)
 * with INT8 row-wise weights and a KV cache. Mirrors sjai/export.py:RefEngine.
 * CPU only: there is no GPU/NNAPI path.
 */
class SjEngine(private val model: SjwModel, threads: Int, maxContext: Int) {
    private class Layer(
        val n1: FloatArray, val qkv: QMatrix, val o: QMatrix, val n2: FloatArray,
        val g: QMatrix, val u: QMatrix, val dn: QMatrix,
    )

    private val cfg = model.config
    private val d = cfg.dModel
    private val h = cfg.nHead
    private val hd = d / h
    val maxContext: Int = minOf(maxContext, cfg.maxSeq)

    @Volatile private var pool: ExecutorService? = null

    @Volatile var threads: Int = 1
        private set

    private val half = hd / 2
    private val cosT = FloatArray(cfg.maxSeq * half)
    private val sinT = FloatArray(cfg.maxSeq * half)
    private val kCache = Array(cfg.nLayer) { FloatArray(this.maxContext * d) }
    private val vCache = Array(cfg.nLayer) { FloatArray(this.maxContext * d) }

    var pos = 0
        private set

    private val layers = (0 until cfg.nLayer).map { i ->
        Layer(
            model.vec("blocks.$i.n1.w"), model.mat("blocks.$i.a.qkv.weight"), model.mat("blocks.$i.a.o.weight"),
            model.vec("blocks.$i.n2.w"), model.mat("blocks.$i.m.g.weight"), model.mat("blocks.$i.m.u.weight"),
            model.mat("blocks.$i.m.d.weight"),
        )
    }
    private val emb = model.mat("emb.weight")
    private val normF = model.vec("norm.w")

    init {
        for (p in 0 until cfg.maxSeq) for (i in 0 until half) {
            val inv = 1.0 / cfg.ropeBase.toDouble().pow(2.0 * i / hd)
            cosT[p * half + i] = cos(p * inv).toFloat()
            sinT[p * half + i] = sin(p * inv).toFloat()
        }
        setThreads(threads)
    }

    @Synchronized
    fun setThreads(n: Int) {
        val t = n.coerceAtLeast(1)
        if (t == threads && (pool != null || t == 1)) return
        pool?.shutdown()
        pool = if (t > 1) Executors.newFixedThreadPool(t) else null
        threads = t
    }

    @Synchronized
    fun close() {
        pool?.shutdown()
        pool = null
    }

    fun reset() { pos = 0 }

    private fun matvec(m: QMatrix, x: FloatArray, out: FloatArray) {
        val p = pool
        if (p == null || m.rows < 64) {
            rows(m, x, out, 0, m.rows)
            return
        }
        val n = threads
        val chunk = (m.rows + n - 1) / n
        val tasks = (0 until n).mapNotNull { t ->
            val a = t * chunk
            val b = minOf(m.rows, a + chunk)
            if (a >= b) null else Callable { rows(m, x, out, a, b) }
        }
        p.invokeAll(tasks).forEach { it.get() }
    }

    private fun rows(m: QMatrix, x: FloatArray, out: FloatArray, from: Int, to: Int) {
        val cols = m.cols
        val q = m.q
        for (r in from until to) {
            var s = 0f
            val base = r * cols
            for (c in 0 until cols) s += q[base + c] * x[c]
            out[r] = s * m.scale[r]
        }
    }

    private fun rms(x: FloatArray, w: FloatArray, out: FloatArray) {
        var ss = 0f
        for (v in x) ss += v * v
        val inv = 1f / sqrt(ss / x.size + 1e-6f)
        for (i in x.indices) out[i] = x[i] * inv * w[i]
    }

    private fun rope(v: FloatArray, off: Int, p: Int) {
        for (head in 0 until h) for (i in 0 until half) {
            val a = off + head * hd + 2 * i
            val c = cosT[p * half + i]
            val s = sinT[p * half + i]
            val x1 = v[a]
            val x2 = v[a + 1]
            v[a] = x1 * c - x2 * s
            v[a + 1] = x1 * s + x2 * c
        }
    }

    /** Feeds one token and returns next-token logits. Throws if the context is full. */
    fun step(token: Int): FloatArray {
        check(pos < maxContext) { "context full" }
        require(token in 0 until cfg.vocabSize) { "token out of range" }
        val x = FloatArray(d)
        val eb = token * emb.cols
        for (i in 0 until d) x[i] = emb.q[eb + i] * emb.scale[token]
        val xn = FloatArray(d)
        val qkv = FloatArray(3 * d)
        val att = FloatArray(d)
        val tmp = FloatArray(d)
        val gate = FloatArray(cfg.dFf)
        val up = FloatArray(cfg.dFf)
        val scores = FloatArray(pos + 1)
        val scale = 1f / sqrt(hd.toFloat())
        for ((li, L) in layers.withIndex()) {
            rms(x, L.n1, xn)
            matvec(L.qkv, xn, qkv)
            rope(qkv, 0, pos)
            rope(qkv, d, pos)
            System.arraycopy(qkv, d, kCache[li], pos * d, d)
            System.arraycopy(qkv, 2 * d, vCache[li], pos * d, d)
            val kc = kCache[li]
            val vc = vCache[li]
            for (head in 0 until h) {
                var mx = -Float.MAX_VALUE
                for (t in 0..pos) {
                    var s = 0f
                    for (i in 0 until hd) s += qkv[head * hd + i] * kc[t * d + head * hd + i]
                    s *= scale
                    scores[t] = s
                    if (s > mx) mx = s
                }
                var sum = 0f
                for (t in 0..pos) {
                    scores[t] = exp(scores[t] - mx)
                    sum += scores[t]
                }
                for (i in 0 until hd) {
                    var a = 0f
                    for (t in 0..pos) a += scores[t] * vc[t * d + head * hd + i]
                    att[head * hd + i] = a / sum
                }
            }
            matvec(L.o, att, tmp)
            for (i in 0 until d) x[i] += tmp[i]
            rms(x, L.n2, xn)
            matvec(L.g, xn, gate)
            matvec(L.u, xn, up)
            for (i in gate.indices) gate[i] = gate[i] / (1f + exp(-gate[i])) * up[i]
            matvec(L.dn, gate, tmp)
            for (i in 0 until d) x[i] += tmp[i]
        }
        rms(x, normF, xn)
        val logits = FloatArray(cfg.vocabSize)
        matvec(emb, xn, logits)
        pos++
        return logits
    }
}
