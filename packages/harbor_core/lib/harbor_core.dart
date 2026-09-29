// Harbor Core — the portable model runtime shared by the Android app, the update
// server, and the Flutter Web dashboard.
//
// The package is deliberately Flutter-free and `dart:io`-free so the identical
// code runs in three places:
//
//  * Android, compiled ahead of time to native code, where it trains on the
//    user's own device;
//  * the Dart VM update server, where it answers `/api/v1/chat`;
//  * the browser, compiled to JavaScript by dart2js, where the admin
//    dashboard's chat pane runs a model without any server at all.
//
// Import this barrel rather than `src/` paths: the internal layout is not part
// of the contract, and the barrel is what keeps the three builds honest about
// depending on the same surface.

// ---- Chat ----------------------------------------------------------------
export 'src/chat/chat_engine.dart';
export 'src/chat/chat_message.dart';
export 'src/chat/tool_protocol.dart';

// ---- Corpus pipeline -----------------------------------------------------
export 'src/corpus/corpus_document.dart';
export 'src/corpus/corpus_store.dart';
export 'src/corpus/device_text.dart';
export 'src/corpus/html_text.dart';
export 'src/corpus/web_collector.dart';

// ---- Model runtime -------------------------------------------------------
export 'src/model/blocks.dart';
export 'src/model/encoding.dart';
export 'src/model/interfaces.dart';
export 'src/model/linalg.dart';
export 'src/model/tiny_lm.dart';
export 'src/model/trainable_model.dart';

// ---- Tokenizer -----------------------------------------------------------
export 'src/tokenizer/byte_tokenizer.dart';

// ---- Tools ---------------------------------------------------------------
export 'src/tools/device_tools.dart';
export 'src/tools/tool.dart';
export 'src/tools/tool_registry.dart';
export 'src/tools/web_tools.dart';

// ---- Training ------------------------------------------------------------
export 'src/training/adamw.dart';
export 'src/training/training_session.dart';