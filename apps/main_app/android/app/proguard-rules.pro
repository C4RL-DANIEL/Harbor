# Harbor — R8 / ProGuard rules for the release build.
#
# The Flutter Gradle plugin already appends
#   <flutter_sdk>/packages/flutter_tools/gradle/flutter_proguard_rules.pro
# (which only carries `-dontwarn io.flutter.plugin.**` and `-dontwarn android.**`)
# and also appends this file automatically. The rules below are therefore purely
# additive and exist so that aggressive shrinking cannot strip anything the
# Flutter embedding or the OTA plugin reaches at runtime.

# ---------------------------------------------------------------------------
# Flutter embedding
# ---------------------------------------------------------------------------
# The Flutter engine calls back into the Java embedding layer from JNI and via
# the generated GeneratedPluginRegistrant, and plugin registration happens
# reflectively; R8 running in full mode (the AGP 8 default) would otherwise
# rename or delete these entry points. Keeping the whole io.flutter surface is
# coarse but is the configuration Flutter itself documents for release builds.
-keep class io.flutter.app.** { *; }
-keep class io.flutter.embedding.** { *; }
-keep class io.flutter.plugin.** { *; }
-keep class io.flutter.plugins.** { *; }
-keep class io.flutter.util.** { *; }
-keep class io.flutter.view.** { *; }
-keep class io.flutter.** { *; }

# Platform-channel messages are decoded into @Keep-annotated generated classes.
-keep @androidx.annotation.Keep class * { *; }
-keepclassmembers class * {
    @androidx.annotation.Keep *;
}

# ---------------------------------------------------------------------------
# Android framework / AndroidX entry points
# ---------------------------------------------------------------------------
# These classes are instantiated by the framework from the merged manifest, or
# discovered through manifest metadata, so R8 cannot see the references.
-keep class * extends android.app.Activity { *; }
-keep class * extends android.app.Application { *; }
-keep class * extends android.app.Service { *; }
-keep class * extends android.content.BroadcastReceiver { *; }
-keep class * extends android.content.ContentProvider { *; }

# The OTA install flow resolves this provider by authority at runtime through
# FileProvider.getUriForFile(); keep it and its FileProvider subclass chain.
-keep class androidx.core.content.FileProvider { *; }
-keep class * extends androidx.core.content.FileProvider { *; }

# AndroidX App Startup discovers InitializationProvider entries from manifest
# metadata and instantiates them reflectively.
-keep class androidx.startup.** { *; }

# ---------------------------------------------------------------------------
# ota_update plugin (sk.fourq.otaupdate)
# ---------------------------------------------------------------------------
# Driven from Dart over method/event channels and from the manifest; it is not
# statically referenced from app code. It extends FileProvider
# (OtaUpdateFileProvider) and uses OkHttp + commons-codec for the streaming
# download and SHA-256 validation, so keep every class and member.
-keep class sk.fourq.otaupdate.** { *; }
-keepclassmembers class sk.fourq.otaupdate.** { *; }

# The plugin's streaming download/checksum stack.
-keep class org.apache.commons.codec.** { *; }
-dontwarn org.apache.commons.codec.**
-dontwarn okhttp3.**
-dontwarn okio.**

# ---------------------------------------------------------------------------
# Kotlin
# ---------------------------------------------------------------------------
# Preserve annotations/signatures and the Kotlin metadata annotation so Kotlin
# reflection and libraries that read @Metadata keep working after obfuscation.
-keepattributes *Annotation*
-keepattributes Signature
-keepattributes InnerClasses
-keepattributes EnclosingMethod
-keepattributes RuntimeVisibleAnnotations
-keepattributes RuntimeVisibleParameterAnnotations
-keepattributes AnnotationDefault
-keepattributes SourceFile,LineNumberTable
-keep class kotlin.Metadata { *; }
-keep class kotlin.jvm.internal.** { *; }

# Coroutines resolve their main dispatcher and exception handlers reflectively.
-keepnames class kotlinx.coroutines.internal.MainDispatcherFactory {}
-keepnames class kotlinx.coroutines.CoroutineExceptionHandler {}
-keepclassmembers class kotlinx.coroutines.** {
    volatile <fields>;
}
-dontwarn kotlinx.coroutines.**

# ---------------------------------------------------------------------------
# Reflection-based plugin models (Gson/@SerializedName)
# ---------------------------------------------------------------------------
# Several Flutter plugins ship Gson-annotated DTOs whose field names are read
# reflectively from JSON; keep those fields even though they look unused.
-keepclassmembers,allowobfuscation class * {
    @com.google.gson.annotations.SerializedName <fields>;
}
-dontwarn com.google.gson.**

# ---------------------------------------------------------------------------
# Flutter deferred components / Play Core
# ---------------------------------------------------------------------------
# The embedding references Play Core's split-install classes, which are absent
# unless the app actually depends on Play Core. Keep them if present and do not
# fail the build if they are missing.
-keep class com.google.android.play.core.** { *; }
-dontwarn com.google.android.play.core.**
-keep class io.flutter.embedding.engine.deferredcomponents.** { *; }
-dontwarn io.flutter.embedding.engine.deferredcomponents.**

# ---------------------------------------------------------------------------
# Optional annotations that may be missing from the runtime classpath.
# ---------------------------------------------------------------------------
-dontwarn javax.annotation.**
-dontwarn org.jetbrains.annotations.**