package com.sjfoundation.ai.engine

import org.json.JSONObject
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.MessageDigest

data class ModelConfig(
    val vocabSize: Int, val nLayer: Int, val nHead: Int, val dModel: Int,
    val dFf: Int, val maxSeq: Int, val ropeBase: Float,
)

/** Row-major int8 matrix with per-row fp32 scales. */
class QMatrix(val rows: Int, val cols: Int, val q: ByteArray, val scale: FloatArray)

class SjwModel(
    val config: ModelConfig,
    val quant: String,
    private val mats: Map<String, QMatrix>,
    private val vecs: Map<String, FloatArray>,
) {
    fun mat(name: String): QMatrix = mats[name] ?: error("missing matrix $name")
    fun vec(name: String): FloatArray = vecs[name] ?: error("missing vector $name")
}

class ModelFormatException(msg: String) : Exception(msg)

object SjwLoader {
    private fun hex(b: ByteArray) = b.joinToString("") { "%02x".format(it) }

    fun load(bytes: ByteArray): SjwModel {
        if (bytes.size < 8 || String(bytes, 0, 4, Charsets.US_ASCII) != "SJW1") throw ModelFormatException("bad magic")
        val hl = ByteBuffer.wrap(bytes, 4, 4).order(ByteOrder.LITTLE_ENDIAN).int
        if (hl <= 0 || 8L + hl > bytes.size) throw ModelFormatException("bad header length")
        val header = JSONObject(String(bytes, 8, hl, Charsets.UTF_8))
        val dataStart = 8 + hl
        val md = MessageDigest.getInstance("SHA-256")
        md.update(bytes, dataStart, bytes.size - dataStart)
        if (hex(md.digest()) != header.getString("data_sha256")) throw ModelFormatException("model integrity check failed")

        val c = header.getJSONObject("config")
        val cfg = ModelConfig(
            c.getInt("vocab_size"), c.getInt("n_layer"), c.getInt("n_head"), c.getInt("d_model"),
            c.getInt("d_ff"), c.getInt("max_seq"), c.getDouble("rope_base").toFloat(),
        )
        val ts = header.getJSONObject("tensors")
        val mats = HashMap<String, QMatrix>()
        val vecs = HashMap<String, FloatArray>()

        fun floats(name: String): FloatArray {
            val t = ts.getJSONObject(name)
            val out = FloatArray(t.getInt("nbytes") / 4)
            ByteBuffer.wrap(bytes, dataStart + t.getInt("offset"), t.getInt("nbytes"))
                .order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(out)
            return out
        }

        for (name in ts.keys().asSequence().toList()) {
            if (name.endsWith(".scale")) continue
            val t = ts.getJSONObject(name)
            val shape = t.getJSONArray("shape")
            if (t.getString("dtype") == "int8") {
                val rows = shape.getInt(0)
                val cols = shape.getInt(1)
                if (t.getInt("nbytes") != rows * cols) throw ModelFormatException("bad tensor size $name")
                val off = dataStart + t.getInt("offset")
                mats[name] = QMatrix(rows, cols, bytes.copyOfRange(off, off + rows * cols), floats("$name.scale"))
            } else if (shape.length() == 1) {
                vecs[name] = floats(name)
            } else {
                throw ModelFormatException("fp32 matrices are not supported on device: $name")
            }
        }
        return SjwModel(cfg, header.getString("quant"), mats, vecs)
    }
}
