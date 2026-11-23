// LOCAL VERIFICATION ONLY (not part of the APK): minimal org.json stand-in so the pure-Kotlin engine
// can be compiled and run on a plain JVM where the Android/org.json classes are unavailable.
package org.json

class JSONArray(private val items: List<Any?>) {
    fun length() = items.size
    fun get(i: Int) = items[i]
    fun getInt(i: Int) = (items[i] as Number).toInt()
    fun getJSONObject(i: Int) = items[i] as JSONObject
}

class JSONObject(private val m: Map<String, Any?>) {
    constructor(s: String) : this(Parser(s).parseObject())
    fun keys(): Iterator<String> = m.keys.iterator()
    fun getRaw(k: String): Any? = m[k]
    fun getString(k: String) = m[k] as String
    fun getInt(k: String) = (m[k] as Number).toInt()
    fun getLong(k: String) = (m[k] as Number).toLong()
    fun getDouble(k: String) = (m[k] as Number).toDouble()
    fun getJSONObject(k: String) = m[k] as JSONObject
    fun getJSONArray(k: String) = m[k] as JSONArray
}

internal class Parser(private val s: String) {
    private var i = 0
    private fun ws() { while (i < s.length && s[i].isWhitespace()) i++ }
    fun parseObject(): Map<String, Any?> { val v = value(); @Suppress("UNCHECKED_CAST") return (v as JSONObject).let { o -> o.keys().asSequence().associateWith { k -> o.raw(k) } } }
    private fun JSONObject.raw(k: String): Any? = this.getRaw(k)
    fun value(): Any? {
        ws()
        return when (val c = s[i]) {
            '{' -> { i++; val m = LinkedHashMap<String, Any?>(); ws(); if (s[i] == '}') { i++; return JSONObject(m) }
                while (true) { ws(); val k = str(); ws(); i++; m[k] = value(); ws(); if (s[i] == ',') i++ else { i++; break } }
                JSONObject(m) }
            '[' -> { i++; val l = ArrayList<Any?>(); ws(); if (s[i] == ']') { i++; return JSONArray(l) }
                while (true) { l.add(value()); ws(); if (s[i] == ',') i++ else { i++; break } }
                JSONArray(l) }
            '"' -> str()
            't' -> { i += 4; true }
            'f' -> { i += 5; false }
            'n' -> { i += 4; null }
            else -> { val st = i; while (i < s.length && (s[i].isDigit() || s[i] in "+-.eE")) i++; val t = s.substring(st, i)
                if (t.contains('.') || t.contains('e') || t.contains('E')) t.toDouble() else t.toLong() }
        }
    }
    private fun str(): String {
        i++; val sb = StringBuilder()
        while (s[i] != '"') {
            if (s[i] == '\\') { i++; when (s[i]) { 'n' -> sb.append('\n'); 't' -> sb.append('\t'); 'r' -> sb.append('\r'); 'b' -> sb.append('\b'); 'f' -> sb.append('\u000c')
                'u' -> { sb.append(s.substring(i + 1, i + 5).toInt(16).toChar()); i += 4 } else -> sb.append(s[i]) } }
            else sb.append(s[i])
            i++
        }
        i++; return sb.toString()
    }
}
