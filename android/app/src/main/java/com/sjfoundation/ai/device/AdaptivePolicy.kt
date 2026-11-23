package com.sjfoundation.ai.device

enum class Profile { LOW_POWER, BALANCED, HIGH_PERFORMANCE }

data class DeviceInfo(
    val totalRamMb: Long,
    val availRamMb: Long,
    val lowMemory: Boolean,
    val cores: Int,
    val abi: String,
    val thermalStatus: Int,      // PowerManager.THERMAL_STATUS_*: 0 none ... 3 severe, 4 critical ...
    val batteryPct: Int,         // -1 unknown
    val charging: Boolean,
    val powerSave: Boolean,
    val freeStorageMb: Long,
)

data class RuntimeConfig(val profile: Profile, val threads: Int, val maxContext: Int, val maxNewTokens: Int)

sealed class Decision {
    data class Run(val threads: Int, val delayMs: Long) : Decision()
    data class Stop(val reason: String) : Decision()
}

/** Pure logic (no Android deps) so it is unit-testable on the JVM. */
object AdaptivePolicy {
    const val THERMAL_MODERATE = 2
    const val THERMAL_SEVERE = 3
    const val MIN_AVAIL_RAM_MB = 150L

    fun autoProfile(d: DeviceInfo): Profile = when {
        d.totalRamMb < 6000 || d.cores < 6 -> Profile.LOW_POWER
        d.totalRamMb < 10000 -> Profile.BALANCED
        else -> Profile.HIGH_PERFORMANCE
    }

    /** modelMaxSeq is the context length the model was trained with; never exceeded. */
    fun choose(d: DeviceInfo, modelMaxSeq: Int, override: Profile? = null): RuntimeConfig {
        val p = override ?: autoProfile(d)
        val cores = d.cores.coerceAtLeast(1)
        return when (p) {
            Profile.LOW_POWER -> RuntimeConfig(p, minOf(2, cores), minOf(modelMaxSeq, 96), 48)
            Profile.BALANCED -> RuntimeConfig(p, (cores / 2).coerceIn(1, 4), modelMaxSeq, 96)
            Profile.HIGH_PERFORMANCE -> RuntimeConfig(p, (cores - 2).coerceIn(1, 6), modelMaxSeq, modelMaxSeq)
        }
    }

    /** Called before every generated token with a fresh DeviceInfo snapshot. */
    fun adjust(base: RuntimeConfig, d: DeviceInfo): Decision {
        if (d.thermalStatus >= THERMAL_SEVERE) return Decision.Stop("The device is too hot. Generation paused to protect it.")
        if (d.lowMemory || d.availRamMb < MIN_AVAIL_RAM_MB) return Decision.Stop("The device is low on memory. Generation stopped to avoid a crash.")
        var threads = base.threads
        var delay = 0L
        if (d.thermalStatus >= THERMAL_MODERATE) { threads = 1; delay = 40 }
        if (d.powerSave) threads = minOf(threads, 2)
        if (d.batteryPct in 0..14 && !d.charging) { threads = 1; delay = maxOf(delay, 20) }
        return Decision.Run(threads.coerceAtLeast(1), delay)
    }
}
