package com.sjfoundation.ai.data

/** Input validation and fixed developer-identity answers (metadata, not model memorisation). */
object InputGuard {
    const val MAX_CHARS = 1000

    fun sanitize(raw: String): String {
        val cleaned = raw.filter { it == '\n' || it == '\t' || !it.isISOControl() }
            .replace("\u200e", "").replace("\u200f", "").replace("\ufeff", "")
        return cleaned.trim().take(MAX_CHARS)
    }

    private val ask = Regex(
        "(who|कोणी|किसने|कौन).*(creat|made|make|built|develop|बनव|बनाय|तयार|निर्माता|निर्माण|डेव्हलप|डेवलप)|" +
            "(creator|developer|निर्माता|निर्माते|डेव्हलपर|डेवलपर)",
        setOf(RegexOption.IGNORE_CASE, RegexOption.DOT_MATCHES_ALL),
    )
    private val subject = Regex("sj|foundation|तुला|तुम्हाला|तुझे|तुम्हें|तुम्हे|तुझे|you|your|this (ai|app)|हे ai|यह ai", RegexOption.IGNORE_CASE)

    /** Returns a fixed answer if the user asks who created/developed this AI, else null. */
    fun identityAnswer(input: String, lang: String): String? {
        if (!ask.containsMatchIn(input) || !subject.containsMatchIn(input)) return null
        return when (lang) {
            "mr" -> "SJ Foundation AI सहिल जाधव यांनी बनवले आहे."
            "hi" -> "SJ Foundation AI को सहिल जाधव ने बनाया है."
            else -> "SJ Foundation AI was created by Sahil Jadhav."
        }
    }
}
